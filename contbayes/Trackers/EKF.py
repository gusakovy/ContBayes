from tqdm import tqdm
import torch
from torch import Tensor
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from contbayes.BNN.BayesNN import FullCovBNN
from contbayes.Utilities.utils import reduce_tensor, multiple_categorical_cov
from contbayes.Detectors.BayesianDeepSIC import BayesianDeepSIC


class Observation:
    """
    Observation class for maintaining the model's inputs and outputs.

    :param inputs: model inputs
    :param outputs: observed outputs
    """

    def __init__(self, inputs: Tensor | None = None, outputs: Tensor | None = None):
        self.inputs = None if inputs is None else torch.atleast_2d(inputs)
        self.outputs = outputs
        self.num_obs = None if inputs is None else self.inputs.size(0)
        self.out_dim = None if outputs is None else torch.atleast_2d(self.outputs).size(-1)

    def update_observation(self, new_inputs: Tensor | None = None, new_outputs: Tensor | None = None):
        """Setter for inputs and outputs."""

        if new_inputs is not None:
            self.inputs = torch.atleast_2d(new_inputs)
            self.num_obs = self.inputs.size(0)
        if new_outputs is not None:
            if self.num_obs != new_outputs.size(0):
                raise ValueError("Number of outputs does not match the number of inputs")
            self.outputs = new_outputs
            self.out_dim = torch.atleast_2d(new_outputs).size(-1)


class SkipCounter:
    """
    Class to keep track of skipped iterations.

    :param categories: list of categories for the skip counter
    """

    def __init__(self, categories: list | tuple | str):
        self.total_skipped = 0
        self.skip_dict = {category: 0 for category in categories}

    def increment(self, categories: list | tuple | str):
        """
        Increment the skip counter.

        :param categories: category or list of categories to increment
        """

        if isinstance(categories, str):
            categories = [categories]
        for category in categories:
            self.skip_dict[category] += 1
        self.total_skipped += 1

    def reset(self):
        for category in self.skip_dict.keys():
            self.skip_dict[category] = 0
        self.total_skipped = 0


class EKF:
    """
    Tracker class for online training of BayesNN models using EKF.

    :param obs_model: observation model (FullCovBNN)
    :param state_model: state model (scalar)
    :param process_noise_var: process noise variance (scalar)
    :param update_limit: allowed delta of state update
    :param obs_noise_cov: observation noise covariance
    :param obs_reduction: reduction method for the observation model
    """

    def __init__(self,
                 obs_model: FullCovBNN,
                 state_model: float,
                 process_noise_var: float,
                 update_limit: float = 0.1,
                 obs_noise_cov: Tensor | None = None,
                 obs_reduction: str = None):

        self.obs_model = obs_model
        self.num_classes = obs_model.output_dim
        self.state_model = torch.tensor(state_model)
        self.process_noise_var = torch.tensor(process_noise_var)
        self.update_limit = update_limit
        self.obs = Observation()
        if self.obs_model.type == "classifier":
            self.obs_noise_cov = None
        else:
            if obs_noise_cov is None:
                raise ValueError("Missing observation noise covariance matrix")
            self.obs_noise_cov = obs_noise_cov
        self.obs_reduction = obs_reduction
        self.filtered = False
        self.skip_counter = SkipCounter(categories=['S', 'Cholesky', 'Delta', 'Non-positive'])

    def predict_state(self):
        """Predict step for the extended Kalman filter."""

        m = self.obs_model.get_loc().view(-1, 1)
        P = self.obs_model.get_cov()
        loc_pred = self.state_model * m
        cov_pred = (self.state_model ** 2) * P + self.process_noise_var * torch.eye(P.size(0))
        self.obs_model.set_loc(loc_pred)
        self.obs_model.set_cov(cov_pred)
        self.filtered = False

    def predict_outputs(self, inputs: Tensor) -> Tensor:
        """Pass the inputs through the underlying network."""

        return self.obs_model.torch_net(inputs)

    def predict_obs_cov(self, inputs: Tensor):
        raise NotImplementedError

    def update_state(self, callback: callable = None, matrix_operation: str = "LeastSquares"):
        """
        Update step for the extended Kalman filter.

        :param callback: callback function with input variables y_true, y_hat, pred_error (innovation), obs_cov (R),
            obs_model (H), pred_error_cov (S), filtered_mean, filtered_cov.
        :param matrix_operation: which method to use to solve the linear system (Inverse, LeastSquares)
        """

        m = self.obs_model.get_loc().view(-1, 1)
        P = self.obs_model.get_cov()

        match self.obs_model.type:
            case "classifier":
                y_hat = self.predict_outputs(self.obs.inputs)[..., :self.num_classes - 1]
                y_hat = y_hat.reshape(-1, 1)

                y_true = one_hot(self.obs.outputs.flatten(), self.num_classes)[..., :(self.num_classes - 1)]\
                    .type(Tensor).reshape(-1, 1)

                R = multiple_categorical_cov(prob_vector=y_hat,
                                             num_classes=self.num_classes,
                                             reduction=self.obs_reduction,
                                             labels=self.obs.outputs)

            case _:
                y_hat = torch.atleast_2d(self.predict_outputs(self.obs.inputs))
                y_true = self.obs.outputs
                R = self.obs_noise_cov

        H = self.obs_model.jacobian(self.obs.inputs, reduction=self.obs_reduction, labels=self.obs.outputs)

        if self.obs_reduction is not None:
            y_true = reduce_tensor(y_true.view(self.obs.num_obs, self.num_classes - 1),
                                   reduction=self.obs_reduction,
                                   num_classes=self.num_classes,
                                   labels=self.obs.outputs).view(-1, 1)
            y_hat = reduce_tensor(y_hat.view(self.obs.num_obs, self.num_classes - 1),
                                  reduction=self.obs_reduction,
                                  num_classes=self.num_classes,
                                  labels=self.obs.outputs).view(-1, 1)

        e = y_true - y_hat

        S = H @ P @ H.T + R
        if torch.allclose(S, torch.zeros(S.size()), atol=1e-4):
            self.skip_counter.increment('S')
            self.filtered = True
            return

        match matrix_operation:
            case "LeastSquares":
                # Compute filtered mean and cov using least-squares solver
                loc_filtered = m + P @ H.T @ torch.linalg.lstsq(S, e)[0]
                cov_filtered = P - P @ H.T @ torch.linalg.lstsq(S, H @ P)[0]
                cov_filtered = 1 / 2 * (cov_filtered + cov_filtered.T) + 1e-5 * torch.eye(P.size(0))

            case "Inverse":
                K = P @ H.T @ torch.inverse(S)
                loc_filtered = (m + K @ e).view(1, -1)
                cov_filtered = P - K @ H @ P
                cov_filtered = 1 / 2 * (cov_filtered + cov_filtered.T) + 1e-5 * torch.eye(P.size(0))

            case _:
                raise ValueError(f"Unknown solver type: {matrix_operation}, available options are "
                                 f"'LeastSquares', 'Inverse'")

        if callback is not None:
            callback(y_true=y_true, y_hat=y_hat, pred_error=e, obs_cov=R, obs_model=H, pred_error_cov=S,
                     filtered_mean=loc_filtered, filtered_cov=cov_filtered)

        old_norm = torch.linalg.vector_norm(m)
        diff_norm = torch.linalg.vector_norm(m - loc_filtered)
        change = diff_norm / old_norm
        eig, _ = torch.lobpcg(cov_filtered, k=1, largest=False)

        delta_cond = abs(change) < self.update_limit
        pos_def_cond = eig.item() > 1e-6

        if delta_cond and pos_def_cond:
            try:
                self.obs_model.set_cov(cov_filtered)
                self.obs_model.set_loc(loc_filtered)
            except RuntimeError:
                print("It happened")
                self.obs_model.set_loc(m)
                self.obs_model.set_cov(P)
                self.skip_counter.increment('Cholesky')
        else:
            if not delta_cond and not pos_def_cond:
                self.skip_counter.increment(['Delta', 'Non-positive'])
            elif not delta_cond:
                self.skip_counter.increment('Delta')
            else:
                self.skip_counter.increment('Non-positive')

        self.filtered = True

    def run(self, dataloader: DataLoader, callback: callable = None, verbose: bool = True):
        """
        Track model using EKF.

        :param dataloader: Dataloader containing observations. Each batch in dataloader is assumed to be a batch of
            observations for the respective time frame
        :param callback: callback function for tracking progress with three input variables: iteration_num, net,
            inputs, outputs
        :param verbose: whether to print the amount of skipped iterations at the end of tracking
        """

        self.skip_counter.reset()
        for i, data in enumerate(tqdm(dataloader)):
            self.predict_state()
            inputs, outputs = data
            self.obs.update_observation(inputs, outputs)
            self.update_state()
            if callback is not None:
                callback(iteration_num=i, net=self.obs_model, inputs=inputs, outputs=outputs)
        if verbose:
            print(f"Skipped {self.skip_counter.total_skipped}/{len(dataloader)} iterations (" +
                  ', '.join(f'{k}: {v}' for k, v in self.skip_counter.skip_dict.items()) + ")")


class DeepsicEKF(EKF):
    """
    Tracker class for online training of BayesianDeepSIC models using EKF.

    :param state_model: BayesianDeepSIC model.
    :param state_model: state model (scalar)
    :param process_noise_var: process noise variance (scalar)
    :param update_limit: allowed delta of state update
    :param obs_reduction: reduction method for the observation model
    """

    def __init__(self, detector: BayesianDeepSIC,
                 state_model: float,
                 process_noise_var: float,
                 update_limit: float = 0.1,
                 obs_reduction: str = None):

        super().__init__(detector.bnn_block, state_model, process_noise_var, update_limit, None, obs_reduction)
        self.detector = detector

    def _update_block(self, layer_num: int, user_num: int):
        """Perform EKF step on a block of the DeepSIC model."""

        self.obs_model.set_params(self.detector.param_matrix[user_num][layer_num])
        self.predict_state()
        self.update_state()
        self.detector.param_matrix[user_num][layer_num] = self.obs_model.get_params()

    def _update_layer(self, layer_num: int, rx, tx, pred=None) -> Tensor:
        """Perform an EKF step on each block in a layer of the DeepSIC model."""

        inputs = self.detector.pred_and_rx_to_input(layer_num, rx, pred)

        for user_idx in range(self.detector.num_users):
            user_indices = range((self.num_classes - 1) * user_idx, (self.num_classes - 1) * (user_idx + 1))
            user_inputs = inputs[..., [i for i in range(self.detector.rx_size +
                                                        (self.num_classes - 1) * self.detector.num_users)
                                       if i not in user_indices]]
            user_labels = tx[..., user_idx].squeeze()
            self.obs.update_observation(user_inputs, user_labels)
            self._update_block(layer_num=layer_num, user_num=user_idx)

        return self.detector.layer_transition(layer_num, rx, pred)

    def _print_skips(self, dataloader: DataLoader):
        """Print the number of skipped iterations."""

        update_count = len(dataloader) * self.detector.num_layers * self.detector.num_users
        print(f"Skipped {self.skip_counter.total_skipped}/{update_count} iterations (" +
              ', '.join(f'{k}: {v}' for k, v in self.skip_counter.skip_dict.items()) + ")")

    def run(self, dataloader: DataLoader, callback: callable = None, verbose: bool = True):
        """
        Track DeepSIC model using EKF.

        :param dataloader: Dataloader containing observations. Each batch in dataloader is assumed to be a batch of
            observations for the respective time frame
        :param callback: callback function for tracking progress with three input variables: iteration_num, detector,
            inputs, outputs
        :param verbose: whether to print the amount of skipped iterations at the end of tracking
        """

        self.skip_counter.reset()
        for i, batch in enumerate(tqdm(dataloader)):
            rx, labels = batch
            predictions = None
            for layer_idx in range(self.detector.num_layers):
                predictions = self._update_layer(layer_idx, rx, labels, predictions)
            if callback is not None:
                callback(iteration_num=i, detector=self.detector, inputs=rx, outputs=labels)
        if verbose:
            self._print_skips(dataloader)
