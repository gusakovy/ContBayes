import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor


class ConstellationCircleDynamics:
    def __init__(self, n_points: int, noise_var: float, speed: float, energy: float, archive_path=True):
        self.n_points = n_points
        self.points = math.sqrt(energy) * torch.tensor([[math.cos(2 * math.pi * k / self.n_points + math.pi),
                                                         math.sin(2 * math.pi * k / self.n_points + math.pi)]
                                                        for k in range(n_points)])
        self.noise_var = noise_var
        self.speed = speed
        self.energy = energy
        self.time = 0
        self.archive_path = archive_path
        self.points_archive = self.points.unsqueeze(0)

    def generate_samples(self, n: int, advance: bool = True, randomize: bool = False) -> \
            tuple[Tensor, Tensor]:
        """
        :param n: number of samples to generate
        :param advance: advances the model to the next time step if True
        :param randomize: randomizes the samples if True
        :return: n / n_points samples from each constellation point if randomize is false, otherwise n samples from
        random constellation points.
        """

        samples = math.sqrt(self.noise_var) * torch.randn(n, 2)

        if randomize:
            labels = torch.tensor(np.random.choice(a=np.array([0, 1, 2, 3]), size=n), dtype=torch.int64)
        else:
            labels = (torch.arange(n) * self.n_points) // n

        samples += self.points[labels]

        if advance:
            self.advance()

        return samples, labels

    def advance(self):
        self.time += 1
        self.points = math.sqrt(self.energy) * \
            torch.tensor([[math.cos(2 * math.pi * (k / self.n_points + self.speed * self.time) + math.pi),
                           math.sin(2 * math.pi * (k / self.n_points + self.speed * self.time)  + math.pi)]
                          for k in range(self.n_points)])
        if self.archive_path:
            self.points_archive = torch.cat([self.points_archive, self.points.unsqueeze(0)])


def plot_decision_zones(model, channel: ConstellationCircleDynamics, num_samples: int=2000, axis=None):

    inputs, labels = channel.generate_samples(num_samples, advance=False)

    # Create meshgrid
    x_min, x_max = inputs[:, 0].min() - 0.1, inputs[:, 0].max() + 0.1
    y_min, y_max = inputs[:, 1].min() - 0.1, inputs[:, 1].max() + 0.1
    grid_size = max(abs(x_min), abs(y_min), abs(x_max), abs(y_max))
    xx, yy = np.meshgrid(np.arange(-grid_size, grid_size, 0.01),
                         np.arange(-grid_size, grid_size, 0.01))

    # Predict each point on the grid
    grid_tensor = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()
    with torch.no_grad():
        grid_predictions = model.predict(grid_tensor, num_predictions=32).argmax(-1).reshape(xx.shape)

    fig = None
    if axis is None:
        fig, axis = plt.subplots(1, 1)

    # Plot the decision boundary and scatter data points
    contour_colors = ["#1f77b4", "#1f77b4", "#ff7f0e", "#ff7f0e", "#2ca02c", "#2ca02c", "#d62728", "#d62728"]
    axis.contourf(xx, yy, grid_predictions.numpy(), alpha=0.7, colors=contour_colors[:2 * channel.n_points])
    scatter_colors = ["blue", "orange", "green", "red"]
    axis.scatter(inputs[..., 0], inputs[..., 1], edgecolors="k", s=20,
                c=[scatter_colors[int(label)] for label in labels])
    axis.set_xlabel('In Phase Component')
    axis.set_ylabel('Quadrature Component')
    axis.set_xlim([-grid_size, grid_size])
    axis.set_ylim([-grid_size, grid_size])

    return fig, axis
