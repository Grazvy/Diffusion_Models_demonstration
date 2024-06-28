import torch
from enum import Enum
import torchvision.transforms as TF
from IPython.core.display_functions import display
from dataLoader import inverse_transform
from utils1 import make_grid, frames2vid
from PIL import Image


class InferType(Enum):
    VIDEO = 0
    IMAGE = 1
    POINT = 2
    NONE = 3

    @staticmethod
    def get_suffix(inference):
        match inference:
            case InferType.VIDEO:
                return ".mp4"
            case InferType.IMAGE:
                return ".png"
            case InferType.POINT:
                return ".json"


class InferenceLogger:
    def __init__(self, inference=InferType.NONE, save_path=None, nrow=8):
        self.outs = []
        self.inference = inference
        self.nrow = nrow
        self.save_path = save_path + InferType.get_suffix(inference)

    def update(self, x):
        if self.inference == InferType.VIDEO:
            x_inv = inverse_transform(x).type(torch.uint8)
            grid = make_grid(x_inv, nrow=self.nrow, pad_value=255.0).to("cpu")
            ndarr = torch.permute(grid, (1, 2, 0)).numpy()[:, :, ::-1]
            self.outs.append(ndarr)

        elif self.inference == InferType.POINT:
            # todo extract 2d point
            pass

    def save_result(self, x):
        if self.inference == InferType.VIDEO:  # Generate and save video of the entire reverse process.
            frames2vid(self.outs, self.save_path)
            display(Image.fromarray(self.outs[-1][:, :, ::-1]))

        elif self.inference == InferType.POINT:
            # todo write all points into a json file
            pass

        elif self.inference == InferType.IMAGE:  # Display and save the image at the final timestep.
            x = inverse_transform(x).type(torch.uint8)
            grid = make_grid(x, nrow=self.nrow, pad_value=255.0).to("cpu")
            pil_image = TF.functional.to_pil_image(grid)
            pil_image.save(self.save_path, format=self.save_path[-3:].upper())
            display(pil_image)

        # reset
        self.outs.clear()
