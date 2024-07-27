import os
import pickle
import numpy as np
import torch
from torch import Tensor
import torch.utils.data as data
from torch.utils.data import DataLoader


def dataloader_to_generator(data_loader):
    """Convert a DataLoader into a generator that yields batches."""

    for batch in data_loader:
        yield batch

def pickle_save(target_data, directory):
    """
    Save an object as a pickle file in the given directory under the name "results_#" where # is chosen automatically
    to be the smallest unused integer.

    :param target_data: the object to be saved
    :param directory: the directory to save the pickle file
    """

    # Ensure directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Find the next available index
    index = 1
    while os.path.exists(os.path.join(directory, f'results_{index}.pkl')):
        index += 1
    filename = f'results_{index}.pkl'

    # Write data to file with pickle
    with open(os.path.join(directory, filename), 'wb') as f:
        pickle.dump(target_data, f)

    return filename

def prepare_experiment_data(channel, num_pilots: int, num_frames: int, snr: int) -> DataLoader:
    """
    Prepares data for online training experiments.

    :param channel: channel used to generate data
    :param num_pilots: number of pilots transmitted per time frame
    :param num_frames: number of time frames
    :param snr: signal-to-noise ratio for the transmitted data
    """


    label_blocks = torch.zeros(0, 0)
    receive_blocks = torch.zeros(0, 0)
    for frame_idx in range(num_frames):
        rx, labels = prepare_single_batch(channel, num_pilots, frame_idx, snr)
        label_blocks = labels if frame_idx == 0 else torch.cat([label_blocks, labels])
        receive_blocks = rx if frame_idx == 0 else torch.cat([receive_blocks, rx])

    dataset = data.TensorDataset(receive_blocks, label_blocks)
    dataloader = data.DataLoader(dataset, batch_size=num_pilots, shuffle=False)
    return dataloader


def prepare_single_batch(channel, num_samples: int, frame_idx: int, snr: int) -> tuple[Tensor, Tensor]:
    """
    Prepares online training experiment data for a single time frame.

    :param channel: channel used to generate data
    :param num_samples: number of samples to create
    :param frame_idx: time frame index
    :param snr: signal-to-noise ratio for the transmitted data
    """


    num_users = channel.num_users
    num_classes = 2 if channel.modulation_type == 'BPSK' else 4
    labels = torch.tensor(np.random.choice(a=np.array(range(num_classes)),
                                           size=(num_samples, num_users)), dtype=torch.int64)
    rx = channel.transmit(s=labels, snr=snr, frame_ind=frame_idx).T
    if channel.modulation_type != 'BPSK':
        rx = torch.view_as_real(rx)
    rx = rx.reshape(num_samples, -1)

    return rx, labels
