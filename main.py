from src import utils
from src.server import Server

config_dict = {}

if __name__ == "__main__":
    utils.read_conf(config_dict)

    central_server = Server(config_dict)
    central_server.setup()

    test_loss, test_accuracy = central_server.evaluate_global_model()
    print("Test Loss        : ", '{0:.4f}'.format(test_loss))
    print("Test Accuracy    : ", '{0:.4f}'.format(test_accuracy))

    central_server.clients[0].train_model()

    test_loss, test_accuracy = central_server.evaluate_global_model()
    print("Test Loss        : ", '{0:.4f}'.format(test_loss))
    print("Test Accuracy    : ", '{0:.4f}'.format(test_accuracy))





