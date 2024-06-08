import itertools
from experiments.config import Config


def _listify_dict(dictionary: dict):
    """Makes every value in a dictionary a list."""

    for key, value in dictionary.items():
        if not isinstance(value, list):
            dictionary[key] = [value]


def deepsic_parameter_combinations(config: Config) -> list[dict]:
    """
    Generates a dictionary of all possible experiment parameters by taking all combinations of parameters from `config`.

    :param config: Config object containing experiment parameters, with possibly more than one value per parameter.
    :return: List of dictionaries, each representing a single a parameter combination.
    """

    if not isinstance(config.tracking['tracking_method'], list):
        config.tracking['tracking_method'] = [config.tracking['tracking_method']]

    experiments_list = []

    if 'Nothing' in config.tracking['tracking_method']:
        no_tracking_config = config.general | config.channel | config.warm_start | config.deepsic
        _listify_dict(no_tracking_config)
        no_tracking_experiments = [dict(zip(no_tracking_config.keys(), run_config))
                                   for run_config in itertools.product(*no_tracking_config.values())]

        for i, _ in enumerate(no_tracking_experiments):
            no_tracking_experiments[i] = (no_tracking_experiments[i] | {key: '-' for key in config.tracking} |
                                          {key: '-' for key in config.ekf})
            no_tracking_experiments[i]['tracking_method'] = 'Nothing'

        experiments_list = experiments_list + no_tracking_experiments
        config.tracking['tracking_method'].remove('Nothing')

    for tracking_method in config.tracking['tracking_method']:
        tracking_config = dict(config.tracking)
        tracking_config['tracking_method'] = [tracking_method]

        if 'KF' in tracking_method:
            kf_config = (config.general | config.channel | config.warm_start | config.deepsic |
                         tracking_config | config.ekf)
            kf_config['num_epochs'] = '-'
            kf_config['num_batches'] = '-'
            _listify_dict(kf_config)
            kf_experiments = [dict(zip(kf_config.keys(), run_config))
                              for run_config in itertools.product(*kf_config.values())]
            experiments_list = experiments_list + kf_experiments

        else:
            gd_config = config.general | config.channel | config.warm_start | config.deepsic | tracking_config
            _listify_dict(gd_config)
            gd_experiments = [dict(zip(gd_config.keys(), run_config))
                              for run_config in itertools.product(*gd_config.values())]

            for i, _ in enumerate(gd_experiments):
                gd_experiments[i] = gd_experiments[i] | {key: '-' for key in config.ekf}

            experiments_list = experiments_list + gd_experiments

    return experiments_list
