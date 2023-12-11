from tqdm import tqdm
import torch
from torch import Tensor
from torch.nn.functional import one_hot
from contbayes.BayesNN import FullCovBNN
from contbayes.utils import reduce_tensor, multiple_categorical_cov


class Observation:
    def __init__(self, inputs=None, outputs=None):
        self.inputs = inputs if inputs is None else torch.atleast_2d(inputs)
        self.outputs = outputs
        self.num_obs = None if inputs is None else self.inputs.size(0)
        self.out_dim = None if outputs is None else self.outputs.size(-1)

    def allocate_space(self, num_obs, out_dim):
        self.inputs = torch.zeros(num_obs, out_dim)
        self.num_obs = num_obs

    def update_observation(self, new_inputs, new_outputs=None):
        if new_outputs is not None:
            assert torch.atleast_2d(new_inputs).size(0) == new_outputs.size(0)
            self.outputs = new_outputs
            self.out_dim = new_outputs.size(-1)
        self.inputs = torch.atleast_2d(new_inputs)
        self.num_obs = self.inputs.size(0)

    def update_outputs(self, new_outputs):
        assert self.inputs.size(0) == new_outputs.size(0)
        self.outputs = new_outputs
        self.out_dim = new_outputs.size(-1)


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
        if self.obs_model.type != "classifier":
            assert obs_noise_cov is not None
            self.obs_noise_cov = torch.tensor(obs_noise_cov)
        else:
            self.obs_noise_cov = None
        self.obs_reduction = obs_reduction
        self.filtered = False

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
            y_true = reduce_tensor(y_true,
                                   reduction=self.obs_reduction,
                                   num_classes=self.num_classes,
                                   labels=self.obs.outputs)
            y_hat = reduce_tensor(y_hat,
                                  reduction=self.obs_reduction,
                                  num_classes=self.num_classes,
                                  labels=self.obs.outputs)

        e = y_true - y_hat

        S = H @ P @ H.T + R
        if torch.allclose(S, torch.zeros(S.size()), atol=1e-4):
            return

        K = P @ H.T @ torch.inverse(S)

        loc_filtered = (m + K @ e).view(1, -1)
        cov_filtered = P - K @ H @ P
        cov_filtered = 1 / 2 * (cov_filtered + cov_filtered.T) + 1e-5 * torch.eye(P.size(0))

        if callback is not None:
            callback(y_true=y_true, y_hat=y_hat, pred_error=e, obs_cov=R, obs_model=H, pred_error_cov=S, kalman_gain=K,
                     filtered_mean=loc_filtered, filtered_cov=cov_filtered)

        old_norm = torch.linalg.vector_norm(m)
        new_norm = torch.linalg.vector_norm(loc_filtered)
        change = (new_norm - old_norm) / old_norm
        eig, _ = torch.lobpcg(cov_filtered, k=1, largest=False)

        if abs(change) < self.update_limit and eig.item() > 1e-4:
            self.obs_model.set_loc(loc_filtered)
            self.obs_model.set_cov(cov_filtered)

        self.filtered = True

    def predict_obs(self, inputs):
        return self.obs_model.torch_net(inputs)

    def predict_obs_cov(self, inputs):
        raise NotImplementedError

    def run(self, obs_generator, num_iterations, n_obs=1, callback=None):
        for i in tqdm(range(num_iterations)):
            self.predict_state()
            inputs, outputs = obs_generator(n_obs)
            self.obs.update_observation(inputs, outputs)
            self.update_state()
            if callback is not None:
                callback(iteration_num=i, net=self.obs_model, inputs=inputs, outputs=outputs)
