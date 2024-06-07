from abc import ABC, abstractmethod


class Experiment(ABC):
    """
    Base class for online learning experiments.

    :param param_dict: dictionary of experiment parameters
    :param verbose: whether to print progress messages
    """

    def __init__(self, param_dict: dict, verbose: bool = False):
        self.params = param_dict
        self.verbose = verbose
        self.runtime = None
        self._validate_experiment()
        self._init_channel()
        self._init_model()

    @abstractmethod
    def _validate_experiment(self):
        """Validate experiment parameters."""
        pass

    @abstractmethod
    def _init_channel(self):
        """Initiate channel."""
        pass

    @abstractmethod
    def _init_model(self):
        """Initiate model."""
        pass

    @abstractmethod
    def _reset_environment(self):
        """Reset environment."""
        pass

    @abstractmethod
    def _warm_start(self):
        """Warm start the model."""
        pass

    @abstractmethod
    def _track(self):
        """Begin tracking."""
        pass

    @abstractmethod
    def _test(self):
        """Test model."""
        pass

    @abstractmethod
    def _log(self, metrics):
        """Log experiment results."""
        pass

    def run(self):
        """Run experiment."""

        self._reset_environment()
        self._warm_start()
        self._reset_environment()
        self._track()
        metrics = self._test()
        self._log(metrics)
