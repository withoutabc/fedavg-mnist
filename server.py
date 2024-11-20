import copy
import random

from client import Client
from mnist import split_data
from model import CNN


class Server:
    def __init__(self, train_dataset, num_clients=100, num_selected=10, num_rounds=10, epochs_per_client=1):
        self.train_dataset = train_dataset
        self.num_clients = num_clients
        self.num_selected = num_selected
        self.num_rounds = num_rounds
        self.epochs_per_client = epochs_per_client
        self.clients = self.create_clients()
        self.global_model = CNN()

    def create_clients(self):
        client_datasets = split_data(self.train_dataset, self.num_clients)
        clients = [Client(dataset, client_id=i, batch_size=64, epochs=self.epochs_per_client) for i, dataset in
                   enumerate(client_datasets)]
        return clients

    def average_weights(self, weight_list):
        avg_weight = {}
        for key in weight_list[0].keys():
            avg_weight[key] = sum([weight[key].float() for weight in weight_list]) / len(weight_list)
        return avg_weight

    def federated_learning(self):
        for round_idx in range(self.num_rounds):
            print(f'\nRound {round_idx + 1}')
            selected_clients = random.sample(self.clients, self.num_selected)
            local_models = []
            local_losses = []
            local_accuracies = []

            for client in selected_clients:
                local_model, local_loss, local_accuracy = client.train(copy.deepcopy(self.global_model))
                local_models.append(local_model)
                local_losses.append(local_loss)
                local_accuracies.append(local_accuracy)

            new_weights = self.average_weights(local_models)
            self.global_model.load_state_dict(new_weights)

            # 计算全局平均损失和准确率
            avg_loss = sum(local_losses) / len(local_losses)
            avg_accuracy = sum(local_accuracies) / len(local_accuracies)
            print(f'Round {round_idx + 1} - Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.2f}%')

        return self.global_model