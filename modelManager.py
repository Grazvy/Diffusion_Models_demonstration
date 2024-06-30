import os
import torch
import shutil
from uNet import UNet
from fullyConnectedNN import FullyConnectedNeuralNetwork
from utils2 import read_config, write_config, yes_no_prompt, add_tar_suffix


class ModelManager:
    """Helper class to handle the U-Net storage, loading, checkpointing and inference logging"""

    def __init__(self, model_name, model_config=None, checkpoint_name=None, type="UNET"):
        """Load model configurations and apply an existing checkpoint.
        If specified folder does not exist, create folder structure and save given model configurations.

        Args:
        model_name (String): this will be used as an ID to load the current state of your model.
        model_config (dict): the specific configurations for your model.
        checkpoint_name (String): name of an existing checkpoint, which you want to apply.
        type (String): the type of model to be used, current options: "UNET", "FCNN"
        """

        os.makedirs("models", exist_ok=True)

        config_path = os.path.join("models", model_name, "config")
        self.folder_path = os.path.join("models", model_name)
        self.checkpoints_path = os.path.join("models", model_name, "checkpoints")
        self.inference_logs_path = os.path.join(self.folder_path, "inference_logs")
        self.checkpoint = None
        self.type = type

        checkpoint_name = add_tar_suffix(checkpoint_name)

        # if model folder exists
        if os.path.exists(self.folder_path):
            # load configurations from existing model folder
            self.model_config = read_config(config_path)

            if model_config is not None:
                print(f"\033[31mWarning: \033[0mprovided \"model_config\" ignored, because an already "
                      f"existing model folder is being loaded.\n{config_path}.json is used instead.")

            # get most recent checkpoint
            if checkpoint_name is None:
                checkpoints = os.listdir(self.checkpoints_path)

                self.checkpoint = max(checkpoints, key=lambda cp: os.path.getmtime(
                    os.path.join(self.checkpoints_path, cp))) if 0 < len(checkpoints) else None

            else:
                if os.path.exists(os.path.join(self.checkpoints_path, checkpoint_name)):
                    raise ValueError(f"provided checkpoint: {checkpoint_name} not found.")

                self.checkpoint = checkpoint_name

        else:
            # create folder structure
            if model_config is None:
                raise ValueError("model_config is missing / None for initial creation")

            if checkpoint_name is not None:
                print(
                    f"\033[31mWarning: \033[0mprovided \"checkpoint_name\" is ignored, because we are creating a new"
                    f" model folder.\nIf you already want a specific name for a checkpoint, use create_checkpoint.")

            os.makedirs(self.checkpoints_path, exist_ok=True)
            os.makedirs(self.inference_logs_path, exist_ok=True)

            self.model_config = model_config
            write_config(config_path, model_config)

    def get_model(self):
        """Create a model instance, based on the stored model configurations and load
        the checkpoint data on it, if any selected"""

        if self.type == "UNET":
            model = UNet(
                input_channels=self.model_config['IMG_SHAPE'][0],
                output_channels=self.model_config['IMG_SHAPE'][0],
                base_channels=self.model_config['BASE_CH'],
                base_channels_multiples=self.model_config['BASE_CH_MULT'],
                apply_attention=self.model_config['APPLY_ATTENTION'],
                dropout_rate=self.model_config['DROPOUT_RATE'],
                time_multiple=self.model_config['TIME_EMB_MULT'],
            )

        elif self.type == "FCNN":
            model = FullyConnectedNeuralNetwork(input_dim=2)

        else:
            raise ValueError(f"Invalid model type provided during initialisation: {self.type}")


        # load an existing checkpoint
        if self.checkpoint is not None:
            checkpoint_path = os.path.join(self.checkpoints_path, self.checkpoint)
            model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['model'])

        return model

    def update_checkpoint(self, optimizer, scaler, model):
        """Save the model state in the current checkpoint file, or a newly created checkpoint file,
        if none is existing yet."""

        if self.checkpoint is None:
            self.checkpoint = "ckpt.tar"
            print(f"new checkpoint created at: {os.path.join(self.checkpoints_path, self.checkpoint)}")

        checkpoint_dict = {
            "opt": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "model": model.state_dict()
        }
        torch.save(checkpoint_dict, os.path.join(self.checkpoints_path, self.checkpoint))
        del checkpoint_dict

    def create_checkpoint(self, optimizer, scaler, model, checkpoint_name):
        """Create a new checkpoint file, save the current model state in it and select it as current checkpoint."""

        checkpoint_name = add_tar_suffix(checkpoint_name)

        checkpoint_path = os.path.join(self.checkpoints_path, checkpoint_name)
        if checkpoint_name is None:
            raise ValueError("checkpoint_name cannot be None")
        if os.path.exists(checkpoint_path):
            raise ValueError(f"provided path: {checkpoint_path} does already exist, you can load existing checkpoints "
                             f"using load_checkpoint.")

        self.checkpoint = checkpoint_name
        self.update_checkpoint(optimizer, scaler, model)
        print(f"new checkpoint stored at: {checkpoint_path}")

    def load_checkpoint(self, model, checkpoint_name):
        """Load selected checkpoint file into the model and select it as current."""

        checkpoint_name = add_tar_suffix(checkpoint_name)

        checkpoint_path = os.path.join(self.checkpoints_path, checkpoint_name)
        if checkpoint_name is None or not os.path.exists(checkpoint_path):
            raise ValueError(f"provided path: {checkpoint_path} does not exist, you can create non existing "
                             f"checkpoints using create_checkpoint.")

        self.checkpoint = checkpoint_name
        checkpoint_path = os.path.join(self.checkpoints_path, self.checkpoint)
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['model'])
        print(f"loaded checkpoint from: {checkpoint_path}")

    def clear_old_logs(self):
        pass

    def delete_model(self):
        """Deletes the whole model folder and the object itself."""

        answer = yes_no_prompt(f"Do you really want to delete the folder: {self.folder_path}? \n"
                               f"This process is irreversible.")
        if answer == 'y':
            print("Proceeding...")
            shutil.rmtree(self.folder_path)
            del self
            print("Deletion successful.")
        else:
            print("Cancelled.")

    def get_inference_logs_path(self):
        log_ids = [int(log.replace("log_", "")) for log in os.listdir(self.inference_logs_path)]
        last_id = max(log_ids) if 0 < len(log_ids) else 0
        file_name = f"log_{last_id + 1}"
        return os.path.join(self.inference_logs_path, file_name)
