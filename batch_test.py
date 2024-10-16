import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

# Define a simple model for testing
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Initial parameters
initial_batch_size = 32  # Starting batch size
batch_size = initial_batch_size

# Instantiate model and optimizer
model = SimpleCNN().cuda()  # Ensure the model is on the GPU
optimizer = optim.Adam(model.parameters())

while True:
    try:
        # Create DataLoader with current batch size
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

        # Test the current batch size by running one training epoch
        model.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.cuda(), labels.cuda()  # Move to GPU

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Batch size {batch_size} is working.")
        batch_size *= 2  # Double the batch size

    except RuntimeError as e:
        print(f"Batch size {batch_size} caused an error: {e}")
        if 'out of memory' in str(e):
            print("Reducing the batch size.")
            batch_size //= 2  # Reduce batch size to the last successful one
            print(f"Try using batch size: {batch_size}")
        break  # Exit the loop on error

print("Finished testing batch sizes.")
