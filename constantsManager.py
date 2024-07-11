import torch


class ConstantsManager:
    """Manage variance schedule and provide variants such as 1 / sqrt(alpha_t).
    Currently, only the linear variance schedule is specified, but feel free to
    include others as well :)

    Args:
    num_diffusion_timesteps (int): this should correspond the amount of timesteps in you diffusion process.
    device (String): the device to use torch on
    shape_2D (boolean): the main code works with image data, this allows compatibility with 2D tensors
    """

    def __init__(self, num_diffusion_timesteps=1000, device="cpu", shape_2d=False):
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.device = device
        self.adjust_shape = shape_2d

        self.betas = self.create_linear_variance_schedule()
        alphas = 1 - self.betas
        alpha_cumulative = torch.cumprod(alphas, dim=0)
        self.sqrt_alpha_cumulative = torch.sqrt(alpha_cumulative)
        self.one_by_sqrt_alpha = 1. / torch.sqrt(alphas)
        self.sqrt_one_minus_alpha_cumulative = torch.sqrt(1 - alpha_cumulative)

    def create_linear_variance_schedule(self):
        # linear schedule
        scale = 1000 / self.num_diffusion_timesteps
        beta_start = scale * 1e-4
        beta_end = scale * 0.02
        return torch.linspace(
            beta_start,
            beta_end,
            self.num_diffusion_timesteps,
            dtype=torch.float32,
            device=self.device,
        )

    def get(self, element: torch.Tensor, t: torch.Tensor):
        """
        Get value at index position "t" in "element" and
            reshape it to have the same dimension as a batch of images.
        """
        res = element.gather(-1, t)

        if self.adjust_shape:
            return res.reshape((-1, 1))

        return res.reshape((-1, 1, 1, 1))

    def beta_at(self, timesteps: torch.Tensor):
        return self.get(self.betas, timesteps)

    def one_by_sqrt_alpha_at(self, timesteps: torch.Tensor):
        return self.get(self.one_by_sqrt_alpha, timesteps)

    def sqrt_alpha_cumulative_at(self, timesteps: torch.Tensor):
        return self.get(self.sqrt_alpha_cumulative, timesteps)

    def sqrt_one_minus_alpha_cumulative_at(self, timesteps: torch.Tensor):
        return self.get(self.sqrt_one_minus_alpha_cumulative, timesteps)

    def get_betas(self):
        return self.betas