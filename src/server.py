import torch
from torch.utils.data import Subset
from src.client import Client
from src.models import CNN
from src import utils


class Server:
    def __init__(self, config_dict):
        print("\nCreating central server...")
        self.data_loader_dict = {}
        self.config_dict = config_dict
        self.data_path = self.config_dict['data_path']
        self.rand_seed = int(self.config_dict['rand_seed'])
        self.batch_size = int(self.config_dict['batch_size'])
        self.device = self.config_dict['device_type']
        self.num_clients = int(self.config_dict['num_clients'])

        self.model = None
        self.in_channel = int(config_dict['in_channel'])
        self.hidden_channel = int(config_dict['hidden_channel'])
        self.num_hidden = int(config_dict['num_hidden'])
        self.num_classes = int(config_dict['num_classes'])
        self.learning_rate = float(config_dict['learning_rate'])
        self.clients = []
        print("done")

    def setup(self):
        # Create the datasets
        print("\nCreating dataset...")
        self.data_loader_dict = utils.load_data(self.data_path, self.rand_seed, self.batch_size)
        print("done")

        # Create the model
        print("\nCreating global model...")
        self.model = CNN(self.in_channel, self.hidden_channel, self.num_hidden, self.num_classes)
        print("done")

        # Initialize model weights
        print("\nInitializing model weights...")
        utils.init_weights(self.model)
        print("done")

        # Create client
        print("\nCreating client...")
        c = Client()
        c.input_model(model=self.model, device=self.device, learning_rate=self.learning_rate)
        c.input_data(data_partition=self.data_loader_dict['train'])
        self.clients.append(c)
        print("done")

    def evaluate_global_model(self):
        print("\nTesting global model...")

        self.model.eval()
        self.model.to(self.device)

        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels in self.data_loader_dict['test']:
                data, labels = data.float().to(self.device), labels.long().to(self.device)

                outputs = self.model(data)

                test_loss += torch.nn.functional.cross_entropy(outputs, labels).item()

                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()

                if self.device == "cuda":
                    torch.cuda.empty_cache()
        self.model.to("cpu")

        test_loss = test_loss / len(self.data_loader_dict['test'])
        test_accuracy = correct / len(self.data_loader_dict['test_raw_dataset'])

        return test_loss, test_accuracy
