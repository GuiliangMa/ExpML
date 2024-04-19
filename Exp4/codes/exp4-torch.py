import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms



class BPNeuralNet(nn.Module):
    def __init__(self):
        super(BPNeuralNet, self).__init__()
        self.Flatten = nn.Flatten()
        self.fc1 = nn.Linear(3*32*32,1024)
        self.ReLU = nn.ReLU()
        self.fc2 = nn.Linear(1024,10)

    def forward(self, x):
        x = self.Flatten(x)
        x = self.fc1(x)
        x = self.ReLU(x)
        x = self.fc2(x)
        return x

    def predict(self,x):
        x=self.forward(x)
        return torch.nn.functional.softmax(x, dim=1)

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_dir = '../data'
    trainSet = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform)
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=32, shuffle=True, num_workers=2)

    testSet = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=32, shuffle=False, num_workers=2)

    model = BPNeuralNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    epochs = 30

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainLoader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 200 == 199:
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 200:.4f}')
                running_loss = 0.0
        print('Finished Training')

    total = 0
    correct = 0
    with torch.no_grad():
        for data in testLoader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')




# print(torch.cuda.is_available())
# print(torch.cuda.device_count())