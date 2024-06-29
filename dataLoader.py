import torchvision.transforms as TF
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from swissRollLoader import SwissRoll2DLoader
from utils1 import DeviceDataLoader


def get_dataset(dataset_name='MNIST'):
    transform_list = [
        TF.ToTensor(),
        TF.Resize((32, 32), interpolation=TF.InterpolationMode.BICUBIC, antialias=True),
        TF.Lambda(lambda t: (t * 2) - 1)  # Scale between [-1, 1]
    ]

    if dataset_name != "MNIST":
        transform_list.insert(2, TF.RandomHorizontalFlip())

    transforms = TF.Compose(transform_list)

    if dataset_name.upper() == "MNIST":
        dataset = datasets.MNIST(root="data", train=True, download=True, transform=transforms)
        total_data_size_bytes = len(dataset) * dataset[0][0].numpy().nbytes
        total_data_size_mb = total_data_size_bytes / (1024 * 1024)
        print(f"Total available data: {len(dataset)} Images / {total_data_size_mb:.2f} MB ")

    elif dataset_name == "Cifar-10":
        dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transforms)
    elif dataset_name == "Cifar-100":
        dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transforms)
    elif dataset_name == "Flowers":
        dataset = datasets.ImageFolder(root="./flowers", transform=transforms)

    return dataset


def get_dataloader(dataset_name='MNIST',
                   batch_size=32,
                   pin_memory=False,
                   shuffle=True,
                   num_workers=0,
                   device="cpu"
                   ):
    if dataset_name == "SWISS":
        # todo make total samples & noise adjustable
        return SwissRoll2DLoader(200, batch_size, 0.15)
    else:
        dataset = get_dataset(dataset_name=dataset_name)
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                pin_memory=pin_memory,
                                num_workers=num_workers,
                                shuffle=shuffle
                                )
        device_dataloader = DeviceDataLoader(dataloader, device)
        return device_dataloader


def inverse_transform(tensors):
    """Convert tensors from [-1., 1.] to [0., 255.]"""
    return ((tensors.clamp(-1, 1) + 1.0) / 2.0) * 255.0
