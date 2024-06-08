from tqdm import tqdm
import torch
from torch import Tensor
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from contbayes.BNN.BayesNN import FullCovBNN
from contbayes.Detectors.BayesianDeepSIC import BayesianDeepSIC
from contbayes.Trackers.EKF import EKF, SkipCounter
import contbayes.Utilities.math_utils as utils


class SqrtEKF(EKF):
    """Tracker class for online training of BayesNN models using Square-root EKF."""

    def __init__(self,
                 obs_model: FullCovBNN,
                 state_model: float,
                 process_noise_var: float,
                 diag_loading: float = 0.0,
                 update_limit: float = 0.1,
                 obs_noise_cov: Tensor = None,
                 obs_reduction: str = None):

        super().__init__(obs_model, state_model, process_noise_var, diag_loading,
                         update_limit, obs_noise_cov, obs_reduction)
        self.skip_counter = SkipCounter(['QR', 'Delta'])

    def predict_state(self):
        """Does nothing. Square root Kalman filter predict and update steps are combined into one update step."""

        pass

    def predict_outputs(self, inputs: Tensor):
        """Pass the inputs through the underlying network."""

        return self.obs_model.torch_net(inputs)

    def predict_obs_cov(self, inputs: Tensor):
        raise NotImplementedError

    def update_state(self, callback: callable = None, **kwargs):
        """
        Update step of the square root Kalman filter.

        :param callback: callback function with input variables y_true, y_hat, pred_error (innovation), obs_cov (R),
            obs_model (H), filtered_mean, filtered_sqrt_cov.
        """

        m = self.obs_model.get_loc().view(-1, 1)
        L_c = utils.scale_and_tril_to_cholesky(self.obs_model.get_scale(), self.obs_model.get_tril())

        match self.obs_model.type:
            case "classifier":
                y_hat = self.predict_outputs(self.obs.inputs)[..., :self.num_classes - 1]
                y_hat = y_hat.reshape(-1, 1)

                y_true = one_hot(self.obs.outputs.flatten(), self.num_classes)[..., :(self.num_classes - 1)]\
                    .type(Tensor).reshape(-1, 1)

                R = utils.multiple_categorical_cov(prob_vector=y_hat,
                                                   num_classes=self.num_classes,
                                                   reduction=self.obs_reduction,
                                                   labels=self.obs.outputs)
                R = R + self.diag_loading * torch.eye(R.size(0))
                R_c = torch.linalg.cholesky(R)

            case _:
                y_hat = torch.atleast_2d(self.predict_outputs(self.obs.inputs))
                y_true = self.obs.outputs
                R = self.obs_noise_cov
                R_c = torch.linalg.cholesky(R)

        H = self.obs_model.jacobian(self.obs.inputs, reduction=self.obs_reduction, labels=self.obs.outputs)

        state_dim = L_c.size(0)
        obs_dim = R_c.size(0)

        if self.obs_reduction is not None:
            y_true = utils.reduce_tensor(y_true.view(self.obs.num_obs, self.num_classes - 1),
                                         reduction=self.obs_reduction,
                                         num_classes=self.num_classes,
                                         labels=self.obs.outputs).view(-1, 1)
            y_hat = utils.reduce_tensor(y_hat.view(self.obs.num_obs, self.num_classes - 1),
                                        reduction=self.obs_reduction,
                                        num_classes=self.num_classes,
                                        labels=self.obs.outputs).view(-1, 1)

        U = torch.cat((torch.cat((H @ L_c, R_c, torch.zeros(obs_dim, state_dim)), dim=1),
                      torch.cat((self.state_model * L_c, torch.zeros(state_dim, obs_dim),
                                 torch.sqrt(self.process_noise_var) * torch.eye(state_dim)), dim=1)), dim=0)
        e = y_true - y_hat

        try:
            U_l = torch.linalg.qr(U.T, mode='r')[1].T
        except RuntimeError:
            self.skip_counter.increment('QR')
            self.filtered = True
            return

        U_l = torch.sign(torch.diag(U_l)) * U_l
        U_1 = U_l[0:obs_dim, 0:obs_dim]
        U_2 = U_l[obs_dim:(obs_dim+state_dim), 0:obs_dim]
        sqrt_cov_filtered = U_l[obs_dim:(obs_dim+state_dim), obs_dim:(obs_dim+state_dim)]

        loc_filtered = self.state_model * m + U_2 @ torch.linalg.solve_triangular(U_1, e, upper=False)

        if callback is not None:
            callback(y_true=y_true, y_hat=y_hat, pred_error=e, obs_cov=R, obs_model=H,
                     filtered_mean=loc_filtered, filtered_sqrt_cov=sqrt_cov_filtered)

        old_norm = torch.linalg.vector_norm(m)
        diff_norm = torch.linalg.vector_norm(m - loc_filtered)
        change = diff_norm / old_norm

        if abs(change) < self.update_limit:
            new_scale, new_tril = utils.cholesky_to_scale_and_tril(sqrt_cov_filtered)
            self.obs_model.set_loc(loc_filtered)
            self.obs_model.set_scale(new_scale)
            self.obs_model.set_tril(new_tril)
        else:
            self.skip_counter.increment('Delta')

    def run(self, dataloader: DataLoader, callback: callable = None, verbose: bool = True):
        """
        Track using square root EKF.

        :param dataloader: Dataloader containing observations. Each batch in dataloader is assumed to be a batch of
            observations for the respective time frame
        :param callback: callback function for tracking progress with input variables iteration_num, net,
            inputs, outputs
        :param verbose: whether to print the amount of skipped iterations at the end of tracking
        """

        self.skip_counter.reset()
        for i, data in enumerate(tqdm(dataloader)):
            inputs, outputs = data
            self.obs.update_observation(inputs, outputs)
            self.update_state()
            if callback is not None:
                callback(iteration_num=i, net=self.obs_model, inputs=inputs, outputs=outputs)
        if verbose:
            print(f"Skipped {self.skip_counter.total_skipped}/{len(dataloader)} iterations (" +
                  ', '.join(f'{k}: {v}' for k, v in self.skip_counter.skip_dict.items()) + ")")


class DeepsicSqrtEKF(SqrtEKF):
    """
    Tracker class for online training of BayesianDeepSIC models using square root EKF.

    :param state_model: BayesianDeepSIC model.
    :param state_model: state model (scalar)
    :param process_noise_var: process noise variance (scalar)
    :param update_limit: allowed delta of state update
    :param obs_reduction: reduction method for the observation model
    """

    def __init__(self, detector: BayesianDeepSIC,
                 state_model: float,
                 process_noise_var: float,
                 diag_loading: float = 0.0,
                 update_limit: float = 0.1,
                 obs_reduction: str = None):

        super().__init__(detector.bnn_block, state_model, process_noise_var, diag_loading,
                         update_limit, None, obs_reduction)
        self.detector = detector

    def _update_block(self, layer_num: int, user_num: int) -> None:
        """Perform square root EKF step on a block of the DeepSIC model."""

        self.obs_model.set_params(self.detector.param_matrix[user_num][layer_num])
        self.update_state()
        self.detector.param_matrix[user_num][layer_num] = self.obs_model.get_params()

    def _update_layer(self, layer_num: int, rx, tx, pred=None) -> Tensor:
        """Perform a square root EKF step on each block in a layer of the DeepSIC model."""

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
        Track DeepSIC model using square root EKF.

        :param dataloader: Dataloader containing observations. Each batch in dataloader is assumed to be a batch of
            observations for the respective time frame
        :param callback: callback function for tracking progress with input variables iteration_num, detector,
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
