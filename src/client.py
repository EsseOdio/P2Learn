import torch


class Client:
    def __init__(self, id_client):
        self.id_client = id_client
        self.model = None
        self.device = None
        self.learning_rate = None
        self.data_partition = None

    def input_model(self, model, device, learning_rate):
        self.model = model
        self.device = device
        self.learning_rate = learning_rate

    def input_data(self, data_partition):
        self.data_partition = data_partition

    def train_model(self):
        print("\nTraining model...")

        self.model.train()
        self.model.to(self.device)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        for e in range(1):
            print("Training client", self.id_client)
            for data, labels in self.data_partition:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                optimizer.zero_grad()
                outputs = self.model(data)

                test_loss = torch.nn.functional.cross_entropy(outputs, labels)

                test_loss.backward()
                optimizer.step()

                if self.device == "cuda":
                    torch.cuda.empty_cache()
        self.model.to("cpu")
