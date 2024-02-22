from tqdm import tqdm
import torch
from torch import Tensor
from torch.nn.functional import one_hot
from contbayes.BNN.BayesNN import FullCovBNN
from contbayes.Utilities.utils import reduce_tensor, multiple_categorical_cov
from contbayes.Trackers.EKF import EKF
from contbayes.Utilities.utils import scale_and_tril_to_cholesky, cholesky_to_scale_and_tril


class SqrtEKF(EKF):
    def __init__(self,
                 obs_model: FullCovBNN,
                 state_model: float,
                 process_noise_var: float,
                 update_limit=0.1,
                 obs_noise_cov: Tensor = None,
                 obs_reduction: str = None):

        super().__init__(obs_model, state_model, process_noise_var, update_limit, obs_noise_cov, obs_reduction)
        self.skip_counter = 0

    def predict_state(self):
        raise NotImplementedError

    def update_state(self, callback=None):

        m = self.obs_model.get_loc().view(-1, 1)
        L_c = scale_and_tril_to_cholesky(self.obs_model.get_scale(), self.obs_model.get_tril())

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
                R = R + 1e-6 * torch.eye(R.size(0))
                R_c = torch.linalg.cholesky(R)

            case _:
                y_hat = torch.atleast_2d(self.predict_obs(self.obs.inputs))
                y_true = self.obs.outputs
                R = self.obs_noise_cov
                R_c = torch.linalg.cholesky(R)

        H = self.obs_model.jacobian(self.obs.inputs, reduction=self.obs_reduction, labels=self.obs.outputs)

        state_dim = L_c.size(0)
        obs_dim = R_c.size(0)

        if self.obs_reduction is not None:
            y_true = reduce_tensor(y_true.view(self.obs.num_obs, self.num_classes - 1),
                                   reduction=self.obs_reduction,
                                   num_classes=self.num_classes,
                                   labels=self.obs.outputs).view(-1, 1)
            y_hat = reduce_tensor(y_hat.view(self.obs.num_obs, self.num_classes - 1),
                                  reduction=self.obs_reduction,
                                  num_classes=self.num_classes,
                                  labels=self.obs.outputs).view(-1, 1)

        U = torch.cat((torch.cat((H @ L_c, R_c, torch.zeros(obs_dim, state_dim)), dim=1),
                      torch.cat((self.state_model * L_c, torch.zeros(state_dim, obs_dim),
                                 torch.sqrt(self.process_noise_var) * torch.eye(state_dim)), dim=1)), dim=0)
        e = y_true - y_hat

        U_l = torch.linalg.qr(U.T, mode='r')[1].T
        U_l = torch.sign(torch.diag(U_l)) * U_l
        U_1 = U_l[0:obs_dim, 0:obs_dim]
        U_2 = U_l[obs_dim:(obs_dim+state_dim), 0:obs_dim]
        sqrt_cov_filtered = U_l[obs_dim:(obs_dim+state_dim), obs_dim:(obs_dim+state_dim)]

        loc_filtered = self.state_model * m + U_2 @ (torch.inverse(U_1) @ e)

        if callback is not None:
            callback(y_true=y_true, y_hat=y_hat, pred_error=e, obs_cov=R, obs_model=H, pred_error_cov=S,
                     filtered_mean=loc_filtered, filtered_sqrt_cov=sqrt_cov_filtered)

        old_norm = torch.linalg.vector_norm(m)
        new_norm = torch.linalg.vector_norm(loc_filtered)
        change = (new_norm - old_norm) / old_norm

        if abs(change) < self.update_limit:
            new_scale, new_tril = cholesky_to_scale_and_tril(sqrt_cov_filtered)
            self.obs_model.set_loc(loc_filtered)
            self.obs_model.set_scale(new_scale)
            self.obs_model.set_tril(new_tril)
        else:
            self.skip_counter += 1

    def predict_obs(self, inputs):
        return self.obs_model.torch_net(inputs)

    def predict_obs_cov(self, inputs):
        raise NotImplementedError

    def run(self, dataloader, callback=None, verbose=True):
        self.skip_counter = 0
        for i, data in enumerate(tqdm(dataloader)):
            inputs, outputs = data
            self.obs.update_observation(inputs, outputs)
            self.update_state()
            if callback is not None:
                callback(iteration_num=i, net=self.obs_model, inputs=inputs, outputs=outputs)
        if verbose:
            print(f"skipped {self.skip_counter}/{len(dataloader)} iterations")
