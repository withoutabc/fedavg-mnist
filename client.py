import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import copy

class Client:
    def __init__(self, dataset, client_id, batch_size=64, epochs=1):
        self.dataset = dataset
        self.client_id = client_id
        self.batch_size = batch_size
        self.epochs = epochs
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def train(self, model):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # 创建学习率调度器
        scheduler = MultiStepLR(optimizer, milestones=[self.epochs // 2], gamma=0.1)

        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for epoch in range(self.epochs):
            for images, labels in self.dataloader:
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # 在每个epoch结束时更新学习率
            scheduler.step()

        accuracy = 100 * correct / total
        average_loss = total_loss / len(self.dataloader)

        print(f'Client {self.client_id} - Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%')

        return copy.deepcopy(model.state_dict()), average_loss, accuracy