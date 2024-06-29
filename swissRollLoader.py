import torch
import numpy as np


class SwissRoll2DLoader:
    def __init__(self, n_samples, batch_size, noise=0.0):
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.noise = noise
        self._generate_data()
        self.indices = np.arange(self.n_samples)
        np.random.shuffle(self.indices)

    def _generate_data(self):
        t = 1.5 * np.pi * (1 + 2 * np.random.rand(self.n_samples))
        x = t * np.cos(t)
        y = t * np.sin(t)
        self.data = np.vstack((x, y)).T
        if self.noise > 0.0:
            self.data += self.noise * np.random.randn(self.n_samples, 2)

        # Normalize the data to be in range [-1, 1]
        self.data = (self.data - self.data.min(axis=0)) / (self.data.max(axis=0) - self.data.min(axis=0)) * 2 - 1
        self.data = torch.tensor(self.data, dtype=torch.float32)

    def __iter__(self):
        self.current_index = 0
        np.random.shuffle(self.indices)  # Shuffle indices at the start of each epoch
        return self

    def __next__(self):
        if self.current_index >= self.n_samples:
            raise StopIteration
        batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        batch_data = self.data[batch_indices]
        batch_data = batch_data.view(-1, 1, 1, 2)  # Reshape to [batch_size, 1, 1, 2]
        self.current_index += self.batch_size
        return batch_data, None

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size
