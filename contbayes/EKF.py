from tqdm import tqdm
import torch
from torch import Tensor
from contbayes.BayesNN import FullCovBNN
from contbayes.utils import categorical_cov


class EKF:
    def __init__(self,
                 obs_model: FullCovBNN,
                 state_model: float,
                 process_noise_var: float,
                 obs_noise_cov: Tensor = None):
        self.obs_model = obs_model
        self.num_classes = obs_model.output_dim
        self.state_model = torch.tensor(state_model)
        self.process_noise_var = torch.tensor(process_noise_var)
        if self.obs_model.type != "classifier":
            assert obs_noise_cov is not None
            self.obs_noise_cov = torch.tensor(obs_noise_cov)
        else:
            self.obs_noise_cov = None

    def predict_state(self):
        m = self.obs_model.get_loc().view(-1, 1)
        P = self.obs_model.get_cov()
        loc_pred = self.state_model * m
        cov_pred = (self.state_model ** 2) * P + self.process_noise_var * torch.eye(P.size(0))
        cov_pred = 1 / 2 * (cov_pred + cov_pred.T)
        self.obs_model.set_loc(loc_pred)
        self.obs_model.set_cov(cov_pred)

    def update_state(self, inputs, outputs, callback=None):

        torch.atleast_2d(inputs, outputs)
        assert inputs.size(0) == outputs.size(0)
        n_obs = inputs.size(0)

        m = self.obs_model.get_loc().view(-1, 1)
        P = self.obs_model.get_cov()

        match self.obs_model.type:
            case "classifier":
                y_hat = self.predict_obs(inputs)[..., :self.num_classes - 1]
                y_hat = y_hat.reshape(-1, 1)

                y_true = torch.zeros(n_obs * (self.num_classes - 1), 1)
                R = torch.zeros(n_obs * (self.num_classes - 1), n_obs * (self.num_classes - 1))
                for obs_idx, label in enumerate(outputs):
                    if label < self.num_classes - 1:
                        y_true[obs_idx * (self.num_classes - 1) + label] = 1
                    R[obs_idx * (self.num_classes - 1): (obs_idx + 1) * (self.num_classes - 1),
                      obs_idx * (self.num_classes - 1): (obs_idx + 1) * (self.num_classes - 1)] = \
                        categorical_cov(y_hat[obs_idx * (self.num_classes - 1): (obs_idx + 1) * (self.num_classes - 1)])
            case _:
                y_hat = torch.atleast_2d(self.predict_obs(inputs))
                y_true = outputs
                R = self.obs_noise_cov

        e = y_true - y_hat

        H = self.obs_model.jacobian(inputs)

        S = H @ P @ H.T + R
        if torch.allclose(S, torch.zeros(S.size()), atol=1e-4):
            return

        K = P @ H.T @ torch.inverse(S)

        loc_filtered = (m + K @ e).view(1, -1)
        cov_filtered = P - K @ H @ P
        cov_filtered = 1 / 2 * (cov_filtered + cov_filtered.T) + 1e-6 * torch.eye(P.size(0))

        if callback is not None:
            callback(y_true=y_true, y_hat=y_hat, pred_error=e, obs_cov=R, obs_model=H, pred_error_cov=S, Kalman_gain=K,
                     filtered_mean=loc_filtered, filtered_cov=cov_filtered)

        if torch.linalg.vector_norm(loc_filtered) / torch.linalg.vector_norm(m) < 1:
            self.obs_model.set_loc(loc_filtered)
            self.obs_model.set_cov(cov_filtered)

    def predict_obs(self, inputs):
        return self.obs_model.torch_net(inputs)

    def predict_obs_cov(self, inputs):
        raise NotImplementedError

    def run(self, obs_generator, num_iterations, n_obs=1, callback=None):
        for i in tqdm(range(num_iterations)):
            self.predict_state()
            inputs, outputs = obs_generator(n_obs)
            self.update_state(inputs, outputs, callback)
            if callback is not None:
                callback(iteration_num=i, obs_model=self.obs_model)
