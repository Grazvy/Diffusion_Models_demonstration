import torch
import torchvision.transforms as TF
from IPython.core.display_functions import display
from dataLoader import inverse_transform
from utils1 import make_grid, frames2vid
from PIL import Image


class InferenceLogger:
    def __init__(self, generate_video=False, save_path=None, nrow=8):
        self.outs = []
        self.generate_video = generate_video
        self.nrow = nrow
        self.save_path = save_path

    def update(self, x):
        if self.generate_video:
            x_inv = inverse_transform(x).type(torch.uint8)
            grid = make_grid(x_inv, nrow=self.nrow, pad_value=255.0).to("cpu")
            ndarr = torch.permute(grid, (1, 2, 0)).numpy()[:, :, ::-1]
            self.outs.append(ndarr)

    def save_result(self, x):
        if self.generate_video:  # Generate and save video of the entire reverse process.
            frames2vid(self.outs, self.save_path)
            display(Image.fromarray(self.outs[-1][:, :, ::-1]))

        else:  # Display and save the image at the final timestep of the reverse process.
            x = inverse_transform(x).type(torch.uint8)
            grid = make_grid(x, nrow=self.nrow, pad_value=255.0).to("cpu")
            pil_image = TF.functional.to_pil_image(grid)
            pil_image.save(self.save_path, format=self.save_path[-3:].upper())
            display(pil_image)
