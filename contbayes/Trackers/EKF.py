from tqdm import tqdm
import torch
from torch import Tensor
from torch.nn.functional import one_hot
from contbayes.BNN.BayesNN import FullCovBNN
from contbayes.Utilities.utils import reduce_tensor, multiple_categorical_cov
from contbayes.Detectors.BayesianDeepSIC import BayesianDeepSIC


class Observation:
    def __init__(self, inputs=None, outputs=None):
        self.inputs = None if inputs is None else torch.atleast_2d(inputs)
        self.outputs = outputs
        self.num_obs = None if inputs is None else self.inputs.size(0)
        self.out_dim = None if outputs is None else torch.atleast_2d(self.outputs).size(-1)

    def update_observation(self, new_inputs=None, new_outputs=None):
        if new_inputs is not None:
            self.inputs = torch.atleast_2d(new_inputs)
            self.num_obs = self.inputs.size(0)
        if new_outputs is not None:
            assert self.num_obs == new_outputs.size(0)
            self.outputs = new_outputs
            self.out_dim = torch.atleast_2d(new_outputs).size(-1)


class EKF:
    def __init__(self,
                 obs_model: FullCovBNN,
                 state_model: float,
                 process_noise_var: float,
                 update_limit=0.1,
                 obs_noise_cov: Tensor = None,
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
            assert obs_noise_cov is not None
            self.obs_noise_cov = obs_noise_cov
        self.obs_reduction = obs_reduction
        self.filtered = False
        self.skip_counter = 0

    def predict_state(self):
        m = self.obs_model.get_loc().view(-1, 1)
        P = self.obs_model.get_cov()
        loc_pred = self.state_model * m
        cov_pred = (self.state_model ** 2) * P + self.process_noise_var * torch.eye(P.size(0))
        self.obs_model.set_loc(loc_pred)
        self.obs_model.set_cov(cov_pred)
        self.filtered = False

    def update_state(self, callback=None):

        m = self.obs_model.get_loc().view(-1, 1)
        P = self.obs_model.get_cov()

        match self.obs_model.type:
            case "classifier":
                y_hat = self.predict_obs(self.obs.inputs)[..., :self.num_classes - 1]
                y_hat = y_hat.reshape(-1, 1)

                y_true = one_hot(self.obs.outputs.flatten(), self.num_classes)[..., :(self.num_classes - 1)]\
                    .type(Tensor).reshape(-1, 1)

                R = multiple_categorical_cov(prob_vector=y_hat,
                                             num_classes=self.num_classes,
                                             reduction=self.obs_reduction,
                                             labels=self.obs.outputs)

            case _:
                y_hat = torch.atleast_2d(self.predict_obs(self.obs.inputs))
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
            self.skip_counter += 1
            self.filtered = True
            return

        # Computation of filtered mean and cov using least-squares solver:
        loc_filtered = m + P @ H.T @ torch.linalg.lstsq(S, e)[0]
        cov_filtered = P - P @ H.T @ torch.linalg.lstsq(S, H @ P)[0]
        cov_filtered = 1 / 2 * (cov_filtered + cov_filtered.T) + 1e-5 * torch.eye(P.size(0))

        '''
        # Computation of Kalman gain using inverse:
        K = P @ H.T @ torch.inverse(S)
        loc_filtered = (m + K @ e).view(1, -1)
        cov_filtered = P - K @ H @ P
        cov_filtered = 1 / 2 * (cov_filtered + cov_filtered.T) + 1e-5 * torch.eye(P.size(0))
        '''

        if callback is not None:
            callback(y_true=y_true, y_hat=y_hat, pred_error=e, obs_cov=R, obs_model=H, pred_error_cov=S,
                     filtered_mean=loc_filtered, filtered_cov=cov_filtered)

        old_norm = torch.linalg.vector_norm(m)
        new_norm = torch.linalg.vector_norm(loc_filtered)
        change = (new_norm - old_norm) / old_norm
        eig, _ = torch.lobpcg(cov_filtered, k=1, largest=False)

        if abs(change) < self.update_limit and eig.item() > 1e-6:
            try:
                self.obs_model.set_cov(cov_filtered)
                self.obs_model.set_loc(loc_filtered)
            except:
                print("It happened")
                self.obs_model.set_loc(m)
                self.obs_model.set_cov(P)
                self.skip_counter += 1
        else:
            self.skip_counter += 1

        self.filtered = True

    def predict_obs(self, inputs):
        return self.obs_model.torch_net(inputs)

    def predict_obs_cov(self, inputs):
        raise NotImplementedError

    def run(self, dataloader, callback=None, verbose=True):
        self.skip_counter = 0
        for i, data in enumerate(tqdm(dataloader)):
            self.predict_state()
            inputs, outputs = data
            self.obs.update_observation(inputs, outputs)
            self.update_state()
            if callback is not None:
                callback(iteration_num=i, net=self.obs_model, inputs=inputs, outputs=outputs)
        if verbose:
            print(f"skipped {self.skip_counter}/{len(dataloader)} iterations")


class DeepsicEKF(EKF):
    def __init__(self, detector: BayesianDeepSIC,
                 state_model: float,
                 process_noise_var: float,
                 update_limit=0.1,
                 obs_noise_cov: Tensor = None,
                 obs_reduction: str = None):

        super().__init__(detector.bnn_block, state_model, process_noise_var, update_limit, obs_noise_cov, obs_reduction)
        self.detector = detector

    def _update_block(self, layer_num, user_num) -> None:

        self.obs_model.set_params(self.detector.param_matrix[user_num][layer_num])
        self.predict_state()
        self.update_state()
        self.detector.param_matrix[user_num][layer_num] = self.obs_model.get_params()

    def _update_layer(self, layer_num, rx, tx, pred=None) -> Tensor:
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

    def run(self, dataloader, callback=None, verbose=True):
        self.skip_counter = 0
        for i, batch in enumerate(tqdm(dataloader)):
            rx, labels = batch
            predictions = None
            for layer_idx in range(self.detector.num_layers):
                predictions = self._update_layer(layer_idx, rx, labels, predictions)
            if callback is not None:
                callback(iteration_num=i, detector=self.detector, inputs=rx, outputs=labels)
        if verbose:
            update_count = len(dataloader) * self.detector.num_layers * self.detector.num_users
            print(f"skipped {self.skip_counter}/{update_count} iterations")

    def predict_obs_cov(self, inputs):
        raise NotImplementedError
