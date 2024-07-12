import torch
from dataLoader import get_dataloader, inverse_transform
from swissRollLoader import SwissRoll2DLoader
from utils1 import make_grid
from utils2 import forward_diffusion_copy
from constantsManager import ConstantsManager
import numpy as np
import matplotlib.pyplot as plt


def plot_points(all_data, title, width=5, height=5):
    all_data = np.vstack(all_data)
    plt.figure(figsize=(width, height))
    plt.scatter(all_data[:, 0], all_data[:, 1], c=np.arctan2(all_data[:, 1], all_data[:, 0]), cmap='viridis')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim([-1.33, 1.33])
    plt.ylim([-1.33, 1.33])
    plt.grid(True)
    plt.show()


def visualize_data(dataset_name="MNIST", amount=72, noise=0.15):
    """trigger download of data if not already happened, then display selected amount"""
    if dataset_name == "SWISS":
        loader = SwissRoll2DLoader(amount, amount, noise)
        all_data = []

        for x0, _ in loader:
            all_data.append(x0.view(-1, 2).numpy())  # Flatten for plotting

        plot_points(all_data, '2D Swiss Roll Data (Normalized to [-1, 1])')

    else:
        loader = get_dataloader(dataset_name=dataset_name, batch_size=amount)

        plt.figure(figsize=(12, 6), facecolor='white')

        for b_image, _ in loader:
            b_image = inverse_transform(b_image).cpu()
            grid_img = make_grid(b_image / 255.0, nrow=12, padding=True, pad_value=1, normalize=True)
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.axis("off")
            break


def visualize_forward(dataset_name="MNIST", timesteps=1000, batch_size=6, uniform_steps=False, device='cpu'):
    if timesteps < 22:
        print(f"\033[31mError: This visualisation of images works only for timesteps above 21.\033")
        return

    round_at = 0 if timesteps < 209 else -1

    if batch_size != 6 and dataset_name != "SWISS":
        print(f"\033[31mWarning:\033 Different batch size for image diffusion is not implemented.")
        batch_size = 6

    # create a list of all the specific timesteps to be displayed
    if uniform_steps:
        specific_timesteps = np.linspace(0, timesteps - 1, 12, dtype=int)
    else:
        one_third = timesteps // 3
        two_thirds = timesteps - one_third
        stepsize1 = one_third // 7
        stepsize2 = two_thirds // 5
        first_half = np.arange(0, timesteps, stepsize1)[:7]
        second_half = np.arange(first_half[-1] + stepsize2, timesteps, stepsize2)[:12]
        specific_timesteps = np.concatenate((first_half, second_half))

    if round_at != 0:
        specific_timesteps = np.round(specific_timesteps, round_at)

        if specific_timesteps[-1] <= timesteps:
            specific_timesteps[-1] = timesteps - 1

    constants = ConstantsManager(
        num_diffusion_timesteps=timesteps,
        device=device
    )
    dataloader = get_dataloader(dataset_name=dataset_name,
                                batch_size=batch_size,
                                device=device)

    x0s, _ = next(iter(dataloader))

    if dataset_name == "SWISS":
        visualize_forward_2D_points(constants, specific_timesteps, x0s)

    else:
        visualize_forward_images(constants, specific_timesteps, x0s)


def visualize_forward_2D_points(constants, specific_timesteps, x0s):
    noisy_images = []

    for timestep in specific_timesteps:
        timestep = torch.as_tensor(timestep, dtype=torch.long)

        xts, _ = forward_diffusion_copy(constants, x0s, timestep)
        noisy_images.append(xts)

    # Plotting each timestep
    plt.figure(figsize=(15, 5))
    for i in range(len(specific_timesteps)):
        all_data = np.vstack(noisy_images[i][0][0])
        plt.subplot(1, len(specific_timesteps), i + 1)
        plt.scatter(all_data[:, 0], all_data[:, 1], c=np.arctan2(all_data[:, 1], all_data[:, 0]), cmap='viridis', s=0.5)
        plt.title(f't={specific_timesteps[i]}')
        plt.xlim(-2.33, 2.33)
        plt.ylim(-2.33, 2.33)
        plt.xticks([])  # Remove x-axis ticks for clarity
        plt.yticks([])  # Remove y-axis ticks for clarity
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()


def visualize_forward_images(constants, specific_timesteps, x0s):
    noisy_images = []

    for timestep in specific_timesteps:
        timestep = torch.as_tensor(timestep, dtype=torch.long)

        xts, _ = forward_diffusion_copy(constants, x0s, timestep)
        xts = inverse_transform(xts) / 255.0
        xts = make_grid(xts, nrow=1, padding=1)

        noisy_images.append(xts)

    # Plot and see samples at different timesteps

    _, ax = plt.subplots(1, len(noisy_images), figsize=(10, 5), facecolor='white')

    for i, (timestep, noisy_sample) in enumerate(zip(specific_timesteps, noisy_images)):
        ax[i].imshow(noisy_sample.squeeze(0).permute(1, 2, 0))
        ax[i].set_title(f"t={timestep}", fontsize=8)
        ax[i].axis("off")
        ax[i].grid(False)

    plt.suptitle("Forward Diffusion Process", y=0.9)
    plt.axis("off")
    plt.show()


def visualize_variance_schedule(timesteps):
    constants = ConstantsManager(timesteps, shape_2d=True)

    # get variance schedule
    betas = constants.get_betas()

    # use at most 100 datapoints
    res = betas[::max(timesteps // 100, 1)]

    # Plotting
    plt.figure(figsize=(10, 3))
    plt.plot(res, marker='o', linestyle='-', color='b', markersize=5)
    plt.title('Variance Schedule')
    plt.xlabel('Timestep')
    plt.ylabel('Beta_t')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
