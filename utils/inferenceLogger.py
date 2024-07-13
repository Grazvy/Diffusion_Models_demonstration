import torch
from enum import Enum
import torchvision.transforms as TF
from IPython.core.display_functions import display
from utils.dataLoader import inverse_transform
from utils.utils1 import make_grid, frames2vid
from utils.utils2 import write_json
from utils.visualisations import plot_points, plot_points_sequence
from PIL import Image


class InferType(Enum):
    VIDEO = 0
    IMAGE = 1
    PLOT = 2
    PLOT_PROCESS = 3
    NONE = 4

    @staticmethod
    def get_suffix(inference):
        match inference:
            case InferType.VIDEO:
                return ".mp4"
            case InferType.IMAGE:
                return ".png"
            case InferType.PLOT:
                return ""
            case InferType.PLOT_PROCESS:
                return ""
            case InferType.NONE:
                return ""


class InferenceLogger:
    """
    Displays the results of the sampling process according to InferType.

    Args:
    inference (InferType): saves the resulting image (IMAGE) or the denoising process as a gif (VIDEO),
                           the equivalent for points is PLOT and PLOT_PROCESS respectively
    save_path (String): path where the image, video or points should be saved
    nrow (int): amount of images per row
    """
    def __init__(self, inference=InferType.NONE, save_path=None, nrow=8, plot_size=5):
        self.outs = []
        self.inference = inference
        self.nrow = nrow
        self.plot_size = plot_size
        self.save_path = save_path + InferType.get_suffix(inference)

    def update(self, x):
        if self.inference == InferType.VIDEO:
            x_inv = inverse_transform(x).type(torch.uint8)
            grid = make_grid(x_inv, nrow=self.nrow, pad_value=255.0).to("cpu")
            ndarr = torch.permute(grid, (1, 2, 0)).numpy()[:, :, ::-1]
            self.outs.append(ndarr)

        elif self.inference == InferType.PLOT:
            if self.outs:
                self.outs[-1] = x.view(x.size(0), 2).tolist()
            else:
                self.outs.append(x.view(x.size(0), 2).tolist())

        elif self.inference == InferType.PLOT_PROCESS:
            self.outs.append(x.view(x.size(0), 2).tolist())

    def save_result(self, x):
        if self.inference == InferType.VIDEO:  # Generate and save video of the entire reverse process.
            frames2vid(self.outs, self.save_path)
            display(Image.fromarray(self.outs[-1][:, :, ::-1]))

        elif self.inference == InferType.PLOT:
            write_json(self.save_path, self.outs)
            plot_points(self.outs[-1], "Sampling results", self.plot_size)

        elif self.inference == InferType.PLOT_PROCESS:
            write_json(self.save_path, self.outs)
            plot_points_sequence(self.outs, self.plot_size, frames=len(self.outs), delay=0.3)

        elif self.inference == InferType.IMAGE:  # Display and save the image at the final timestep.
            x = inverse_transform(x).type(torch.uint8)
            grid = make_grid(x, nrow=self.nrow, pad_value=255.0).to("cpu")
            pil_image = TF.functional.to_pil_image(grid)
            pil_image.save(self.save_path, format=self.save_path[-3:].upper())
            display(pil_image)

        # reset
        self.outs.clear()
