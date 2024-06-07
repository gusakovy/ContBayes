import yaml


class Config:
    """
    Config object to hold config.yaml parameters.
    """

    __instance = None

    def __new__(cls, path: str):
        """
        Create new instance of Config from yaml file.

        :param path: path to yaml configuration file
        """

        if Config.__instance is None:
            Config.__instance = object.__new__(cls)
            Config.__instance._load_config(path)
        return Config.__instance

    def _load_config(self, config_path: str):
        """Load attributes from yaml file."""

        with open(config_path) as f:
            config = yaml.safe_load(f)

        for k, v in config.items():
            setattr(self, k, v)
