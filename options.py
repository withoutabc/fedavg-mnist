import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Federated Learning with FedAvg')
    parser.add_argument('--num_clients', type=int, default=100, help='Number of clients')
    parser.add_argument('--num_selected', type=int, default=10, help='Number of clients selected per round')
    parser.add_argument('--num_rounds', type=int, default=10, help='Number of communication rounds')
    parser.add_argument('--epochs_per_client', type=int, default=20, help='Number of epochs per client')

    args = parser.parse_args()
    return args