import numpy as np


class MeanSquaredError:
    def __init__(self):
        pass

    def compute(self, y_pred, y_true):
        """
        computes the mean squared error loss
            args:
                y_pred: predicted labels (n_classes, batch_size)
                y_true: true labels (n_classes, batch_size)
            returns:
                mean squared error loss
        """
        # TODO: Implement mean squared error loss
        # Done
        batch_size = y_pred.shape[1]
        cost = np.sum((y_pred - y_true)**2, axis=1) / batch_size
        return np.squeeze(cost)

    def backward(self, y_pred, y_true):
        """
        computes the derivative of the mean squared error loss
            args:
                y_pred: predicted labels (n_classes, batch_size)
                y_true: true labels (n_classes, batch_size)
            returns:
                derivative of the mean squared error loss
        """
        # TODO: Implement backward pass for mean squared error loss
        # Done
        return np.squeeze(np.sum(2 * (y_pred - y_true), axis=1) / y_pred.shape[1])
