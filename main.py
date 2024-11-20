from mnist import get_mnist_data, test_model
from options import get_args
from server import Server

args = get_args()

train_loader, test_loader = get_mnist_data()

server = Server(train_dataset=train_loader.dataset, num_clients=args.num_clients, num_selected=args.num_selected, num_rounds=args.num_rounds,
                epochs_per_client=args.epochs_per_client)

global_model = server.federated_learning()

test_model(global_model, test_loader)
