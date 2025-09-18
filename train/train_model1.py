import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from tqdm import tqdm

# Dataset
class ThumbDataset(Dataset):
    def __init__(self,data):
        self.X = np.stack(data['Resized_Blobs'].values).astype(np.float16)
        self.X = self.X[:, np.newaxits, :, :] # (N, 1, H, W)

        y = data['Handedness'].values
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(y).astype(np.int8)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
# Load data
df = pd.read_pickle('processed_data_thumb.pkl')
X_train, X_test = train_test_split(df, test_size=0.2, stratify=df['Handedness'], random_state=42)

train_dataset = ThumbDataset(X_train)
test_dataset = ThumbDataset(X_test)

train_loader = DataLoader(train_dataset, batch_size= 64, shuffle= True)
test_loader = DataLoader(test_dataset, batch_size= 64, shuffle = False)

# Model
class ThumbClassifier1(nn.Module):
    def __init__(self):
        super(ThumbClassifier1,self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.25)

        self.flatten = nn.Flatten()
        self.fc3 = nn.Linear(32*4*4, 64)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.25)
        
        self.fc4 = nn.Linear(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)

        return x

# Train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ThumbClassifier1().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)

for epoch in range(20):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        inputs, labels = input.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    acc = correct / total
    print(f"Epoch {epoch+1} Loss: {running_loss:.4f}, Accuracy: {acc:.4f}")

# Eval
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for inputs, labels in test_dataset:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct = predicted.eq(labels).sum().item()
print(f"Test Accuracy: {correct / total:.4f}")

torch.save({'state_dict': model.state_dict()}, 'thumb_classifier1.pth.tar')