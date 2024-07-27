from tqdm import tqdm
from torch.utils.data import DataLoader
from contbayes.Trackers.EKF import SkipCounter


class DeepsicTracker:
    """
    Tracker class for online training of DeepSIC models using SGD or BayesianDeepSIC models using SVI.

    :param detector: DeepSIC model to be trained
    :param num_epochs: number of training epochs per time frame
    :param num_batches: number of training batches per epoch
    """


    def __init__(self, detector, num_epochs, num_batches, learning_rate, update_prior=False):
        self.detector = detector
        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.learning_rate = learning_rate
        self.skip_counter = SkipCounter(categories=[])
        self.update_prior = update_prior

    def run(self, dataloader: DataLoader, callback: callable = None):
        """
        Track model using SGD or SVI.

        :param dataloader: Dataloader containing observations. Each batch in dataloader is assumed to be a batch of
            observations for the respective time frame
        :param callback: callback function for tracking progress with input variables iteration_num, detector,
            inputs, outputs
        """

        for i, pilots in enumerate(tqdm(dataloader)):
            rx, labels = pilots
            batch_size = rx.size(0) // self.num_batches
            self.detector.fit(rx=rx, labels=labels, num_epochs=self.num_epochs, batch_size=batch_size,
                              lr=self.learning_rate, update_prior=self.update_prior)
            if callback is not None:
                callback(iteration_num=i, detector=self.detector, inputs=rx, outputs=labels)
