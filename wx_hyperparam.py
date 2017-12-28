
class WxHyperParameter(object):
    """
    wx feature selector hyperparameters
    """
    def __init__(self, epochs=25, batch_size=10, learning_ratio=0.01, weight_decay=1e-6, momentum=0.9):

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_ratio = learning_ratio
        self.weight_decay = weight_decay
        self.momentum = momentum