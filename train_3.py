import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Grayscale, ToPILImage

imgs_train, points_train = get_train_data()

class LandmarkDataset(Dataset):
    def __init__(self, images, landmarks, transform=None):
        self.images = images
        self.landmarks = landmarks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        landmark = self.landmarks[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(landmark, dtype=torch.float32)

# Transform
transform = Compose([
    ToPILImage(), 
    Grayscale(),
    Resize((96, 96)),
    ToTensor(),  
])

train_dataset = LandmarkDataset(imgs_train, points_train, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

class LandmarkModel(nn.Module):
    def __init__(self):
        super(LandmarkModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(128, 64, kernel_size=1, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 30),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def save_checkpoint(state, filename="checkpoint.pth"):
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}.")

def load_checkpoint(filename, model, optimizer):
    if os.path.isfile(filename):
        print(f"Loading checkpoint from {filename}...")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded. Starting from epoch {epoch}.")
        return epoch
    else:
        print("Checkpoint file not found!")
        return 0

def train_model(model, dataloader, epochs=300, checkpoint_path="checkpoint.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    start_epoch = load_checkpoint(checkpoint_path, model, optimizer)

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        for images, landmarks in dataloader:
            images, landmarks = images.to(device), landmarks.to(device)

            outputs = model(images)
            loss = criterion(outputs, landmarks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader)}")


        save_checkpoint({
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }, checkpoint_path)

if __name__ == "__main__":
    model = LandmarkModel()
    train_model(model, train_loader)