import json
import torch


def forward_diffusion_copy(constants, x0: torch.Tensor, t: torch.Tensor):
    """Copy of forward diffusion, since original is displayed in Notebook"""
    e = torch.randn_like(x0)
    x_t = (constants.sqrt_alpha_cumulative_at(timesteps=t) * x0 + constants.sqrt_one_minus_alpha_cumulative_at(
        timesteps=t) * e)
    return x_t, e


def read_json(file_path):
    with open(file_path + '.json', 'r') as f:
        return json.load(f)


def write_json(file_path, dct):
    with open(file_path + '.json', 'w') as f:
        json.dump(dct, f, indent=4)


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
