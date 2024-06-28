import json
import matplotlib.pyplot as plt
from dataLoader import get_dataloader, inverse_transform
from utils1 import make_grid


def visualize_data(dataset_name="MNIST", amount=72):
    """trigger download of data if not already happened, then display selected amount"""
    loader = get_dataloader(dataset_name=dataset_name, batch_size=amount)

    plt.figure(figsize=(12, 6), facecolor='white')

    for b_image, _ in loader:
        b_image = inverse_transform(b_image).cpu()
        grid_img = make_grid(b_image / 255.0, nrow=12, padding=True, pad_value=1, normalize=True)
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.axis("off")
        break


def read_config(file_path):
    with open(file_path + '.json', 'r') as f:
        return json.load(f)


def write_config(file_path, config):
    with open(file_path + '.json', 'w') as f:
        json.dump(config, f, indent=4)


def yes_no_prompt(message):
    while True:
        response = input(message + " (y/n): ").strip().lower()
        if response == 'y' or response == 'n':
            return response
        else:
            print("Please enter 'y' for yes or 'n' for no.")


def add_tar_suffix(checkpoint_name):
    if checkpoint_name is None:
        return None

    if checkpoint_name.endswith(".tar"):
        return checkpoint_name

    return checkpoint_name + ".tar"
