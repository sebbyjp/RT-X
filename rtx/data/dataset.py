from typing import Optional
from octo.data.dataset import make_interleaved_dataset, make_single_dataset
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights, make_oxe_dataset_kwargs
from octo.data.utils.data_utils import NormalizationType, get_dataset_statistics
from dlimp import DLataset
import torch
import numpy as np
import datasets
from absl import logging
from rtx.data.registry import DATASET_MIXES
from torch.utils.data import IterableDataset, get_worker_info
import torch.distributed as dist
from PIL import Image

import h5py
from torch.utils.data import Dataset
from torchvision import transforms

class HD5Dataset(Dataset):
    def __init__(self, hdf5_file, transform=None, window_size=6, future_action_window_size=0, shuffle=True, seed=42):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.hdf5_file = hdf5_file
        self.transform = transform
        if self.transform is None:
            # Define your transformations / augmentations
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomCrop(224),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            ])
        self.window_size = window_size
        self.future_action_window_size = future_action_window_size
        self.shuffle = shuffle
        self.seed = seed
        self.idxs = np.arange(0, self.length)
        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(self.idxs)


        # Open the hdf5 file and get the size of the dataset
        with h5py.File(self.hdf5_file, 'r') as file:
            self.length = file.attrs['size']

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx == self.length:
            if self.shuffle:
                np.random.seed(self.seed)
                np.random.shuffle(self.idxs)
            idx = 0

        idx = self.idxs[idx]

        observation = np.zeros((self.window_size, 224, 224, 3))
        with h5py.File(self.hdf5_file, 'r') as file:
            if idx - self.window_size < 0:
                observation[-idx:] = np.array([self.transform(Image.fromarray(np.array(observation['observation/image_head'][i]))) for i in range(idx)])
            else:
                observation = np.array([self.transform(Image.fromarray(np.array(observation['observation/image_head'][i]))) for i in range(idx - self.window_size, idx)])
            # Open the hdf5 file and get the sample

        return {'observation': {"image_primary": observation,},
                            #    "image_wrist": sample['observation']['image_wrist']},
            'action': {'x:': file['action/left_hand/x'][idx], 'y': file['action/left_hand/y'][idx], 'z': file['action/left_hand/z'][idx],
                            'yaw': file['action/left_hand/yaw'][idx], 'pitch': file['action/left_hand/pitch'][idx], 'roll': file['action/left_hand/roll'][idx],
                            'grasp': file['action/left_hand/grasp'][idx]},
            'language_instruction': str(file['observation/language_instruction'][idx])}


class TorchRLDSDataset(IterableDataset):
    """Thin wrapper around RLDS dataset for use with PyTorch dataloaders."""

    def __init__(
        self,
        rlds_dataset,
        dataset_statistics,
        sample_weights = None,
        rank=0,
        world_size=1,
        train=True,
    ):
        self._rlds_dataset = rlds_dataset
        self.dataset_statistics = dataset_statistics
        self.sample_weights = sample_weights
        self.rank = rank
        self.world_size = world_size
       
        # if not hasattr(self._rlds_dataset, "dataset_statistics"):
        #     self._rlds_dataset.dataset_statistics = get_dataset_statistics(
        #         self._rlds_dataset
        #     )
        self._is_train = train

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        mod = self.world_size
        shift = self.rank

        if worker_info:
            mod *= worker_info.num_workers
            shift = self.rank * worker_info.num_workers + worker_info.id

        for i, sample in enumerate(self._rlds_dataset.iterator(prefetch=1)):
             if (i + shift) % mod == 0:
                yield {'observation': {"image_primary": sample['observation']['image_primary']},
                                #    "image_wrist": sample['observation']['image_wrist']},
                   'action': sample['action'],
                   'language_instruction': sample['task']['language_instruction'].decode()}

         
    def __len__(self):
        lengths = np.array(
            [
                stats["num_transitions"]
                for stats in self.dataset_statistics
            ]
        )
        if self.sample_weights is not None:
            lengths *= np.array(self.sample_weights)
        total_len = lengths.sum()
        if self._is_train:
            return int(0.95 * total_len)
        else:
            return int(0.05 * total_len)




def get_interleaved_oxe_dataset(mix_name: str = "eef_pose_magic_soup", data_dir: str = "gs://gresearch/robotics", train: bool = True, data_augmentation=True, shuffle_buffer_size=10000) -> DLataset:

    dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
        mix_name,
        data_dir,
        load_camera_views=("primary", "wrist"),
       action_proprio_normalization_type= NormalizationType.NONE,
    )
    logging.info("Creating interleaved OXE dataset {} from {}".format(mix_name, data_dir))
    return make_interleaved_dataset(
        dataset_kwargs_list,
        sample_weights,
        train=train,
        shuffle_buffer_size=shuffle_buffer_size,  # change to 500k for training, large shuffle buffers are important, but adjust to your RAM
        batch_size=None,  # batching will be handles in PyTorch Dataloader object
        balance_weights=True,
        traj_transform_kwargs=dict(
            goal_relabeling_strategy=None,
            window_size=6,
            future_action_window_size=0,
            subsample_length=100,
        ),
        frame_transform_kwargs=dict(
            image_augment_kwargs={
                "primary": dict(
                    random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
                    random_brightness=[0.1],
                    # random_contrast=[0.9, 1.1],
                    # random_saturation=[0.9, 1.1],
                    random_hue=[0.05],
                    augment_order=[
                        "random_resized_crop",
                        "random_brightness",
                        # "random_contrast",
                        # "random_saturation",
                        "random_hue",
                    ],
                ),
            } if data_augmentation else {} ,
            resize_size=dict(
                primary=(224, 224),
                # wrist=(128, 128),
            ),
            num_parallel_calls=200,
        ),
        traj_transform_threads=18,
        traj_read_threads=18,
    )

def get_single_oxe_dataset(name: str = "fractal20220817_data", data_dir: str = "gs://gresearch/robotics", train: bool = True,
data_augmentation=True, shuffle_buffer_size=1000)-> DLataset:
    dataset_kwargs = make_oxe_dataset_kwargs(
    # see octo/data/oxe/oxe_dataset_configs.py for available datasets
    # (this is a very small one for faster loading)
    # "austin_buds_dataset_converted_externally_to_rlds",
    name,
    
    # can be local or on cloud storage (anything supported by TFDS)
    # "/path/to/base/oxe/directory",
    data_dir,
    action_proprio_normalization_type= NormalizationType.NONE,
    )
    logging.info("Creating single OXE dataset {} from {}".format(name, data_dir))
    dataset, dataset_statistics = make_single_dataset(dataset_kwargs, train=train,
      traj_transform_kwargs=dict(
            goal_relabeling_strategy=None,
            window_size=6,
            future_action_window_size=0,
            subsample_length=100,
        ),
           frame_transform_kwargs=dict(
            image_augment_kwargs={
                "primary": dict(
                    random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
                    random_brightness=[0.1],
                    random_contrast=[0.9, 1.1],
                    random_saturation=[0.9, 1.1],
                    random_hue=[0.05],
                    augment_order=[
                        "random_resized_crop",
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                ),
            } if data_augmentation else {},
            resize_size=dict(
                primary=(224, 224),
            ),
            num_parallel_calls=200,
        ),)
    return (dataset.flatten().shuffle(buffer_size=shuffle_buffer_size),[dataset_statistics], None)

def get_oxe_dataset(name: str = "fractal20220817_data", train: bool = True, data_augmentation=True, shuffle_buffer_size=1000) -> (DLataset, list[dict], Optional[dict]) :
    if name in DATASET_MIXES:
        return get_interleaved_oxe_dataset(name, train=train, data_augmentation=data_augmentation, shuffle_buffer_size=shuffle_buffer_size)
    else:
        return get_single_oxe_dataset(name, train=train, data_augmentation=data_augmentation, shuffle_buffer_size=shuffle_buffer_size)

def get_hf_dataset(  
        dataset_path: str = "jxu124/OpenX-Embodiment",
        dataset_name: str = "fractal20220817_data",
        split: str = "train",
        streaming: bool = True):
    logging.info("Fetching dataset {}/{}".format(dataset_path, dataset_name))
    ds = datasets.load_dataset(dataset_path,
                               dataset_name,
                               streaming=streaming,
                               split=split,
                               cache_dir="dataset_cache")
    return ds