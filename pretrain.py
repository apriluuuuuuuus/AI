import os
import shutil
import sys
import torch
import yaml
import numpy as np
from datetime import datetime

# Make `utils` importable when this script is launched as `python scripts/pretrain.py`.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.nt_xent import NTXentLoss


apex_support = False
try:
    sys.path.append("./apex")
    from apex import amp

    apex_support = True
except:
    # Apex is only needed when fp16_precision is enabled. CPU smoke tests set it
    # to false, so this warning is harmless for the reproduction workflow.
    print(
        "Please install apex for mixed precision training from: https://github.com/NVIDIA/apex"
    )
    apex_support = False


def _save_config_file(model_checkpoints_folder):
    # Store the exact config next to checkpoints so a smoke run is reproducible.
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy(args.config, os.path.join(model_checkpoints_folder, "config.yaml"))


class PreTrain(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()

        dir_name = datetime.now().strftime("%b%d_%H-%M-%S")
        log_dir = os.path.join("ckpt", dir_name)
        # TensorBoard logs and checkpoints share this timestamped run folder.
        self.writer = SummaryWriter(log_dir=log_dir)

        self.dataset = dataset
        self.nt_xent_criterion = NTXentLoss(
            self.device, config["batch_size"], **config["loss"]
        )

    def _get_device(self):
        if torch.cuda.is_available() and self.config["gpu"] != "cpu":
            device = self.config["gpu"]
            torch.cuda.set_device(device)
        else:
            device = "cpu"
        print("Running on:", device)

        return device

    def _step(self, model, xis, xjs, n_iter):
        # Each molecule is augmented twice by the dataset. The model embeds both
        # views, and NT-Xent pulls paired views together in representation space.
        ris, zis = model(xis)  # [N,C]

        rjs, zjs = model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.nt_xent_criterion(zis, zjs)
        return loss

    def train(self):
        train_loader, valid_loader = self.dataset.get_data_loaders()

        # The model definition lives in scripts/molea_pretrain.py from upstream.
        from molea_pretrain import MOLEA
        model = MOLEA(**self.config["model"]).to(self.device)
        model = self._load_pre_trained_weights(model)
        print(model)

        optimizer = torch.optim.Adam(
            model.parameters(),
            self.config["init_lr"],
            weight_decay=float(self.config["weight_decay"]),
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config["epochs"] - self.config["warm_up"],
            eta_min=0,
            last_epoch=-1,
        )

        if apex_support and self.config["fp16_precision"]:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level="O2", keep_batchnorm_fp32=True
            )

        model_checkpoints_folder = os.path.join(self.writer.log_dir, "checkpoints")

        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        

        for epoch_counter in range(self.config["epochs"]):
            
            for bn, (xis, xjs) in enumerate(train_loader):
                
                optimizer.zero_grad()

                

                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(model, xis, xjs, n_iter)

                if n_iter % self.config["log_every_n_steps"] == 0:
                    self.writer.add_scalar("train_loss", loss, global_step=n_iter)
                    self.writer.add_scalar(
                        "cosine_lr_decay",
                        scheduler.get_last_lr()[0],
                        global_step=n_iter,
                    )
                    print("Epoch:", epoch_counter, "Iteration:", bn, "Train loss:",loss.item())

                if apex_support and self.config["fp16_precision"]:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1

            

            # validate the model if requested
            if epoch_counter % self.config["eval_every_n_epochs"] == 0:
                valid_loss = self._validate(model, valid_loader)
                print("Epoch:", epoch_counter, "Iteration:", bn, "Valid loss:", valid_loss)
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(
                        model.state_dict(),
                        os.path.join(model_checkpoints_folder, "model.pth"),
                    )

                self.writer.add_scalar(
                    "validation_loss", valid_loss, global_step=valid_n_iter
                )
                valid_n_iter += 1

            if (epoch_counter + 1) % self.config["save_every_n_epochs"] == 0:
                # Save every requested epoch in addition to the best model.pth.
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        model_checkpoints_folder,
                        "model_{}.pth".format(str(epoch_counter)),
                    ),
                )

            # warmup for the first few epochs
            if epoch_counter >= self.config["warm_up"]:
                scheduler.step()

    def _load_pre_trained_weights(self, model):
        # The original script used a hard-coded Linux checkpoint path. For local
        # reproduction, an empty load_model means "train from scratch"; otherwise
        # accept either a direct checkpoint folder or a ckpt/<name>/checkpoints run.
        if not self.config.get("load_model"):
            print("No pre-trained weights configured. Training from scratch.")
            return model

        try:
            load_model = self.config["load_model"]
            if os.path.isdir(load_model):
                checkpoints_folder = load_model
            else:
                checkpoints_folder = os.path.join("ckpt", load_model, "checkpoints")
            print(checkpoints_folder)
            state_dict = torch.load(
                os.path.join(checkpoints_folder, "model.pth"),
                map_location=self.device,
            )
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0
            for (xis, xjs) in valid_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(model, xis, xjs, counter)
                
                valid_loss += loss.item()
                counter += 1
            if counter == 0:
                # Small datasets can create an empty validation loader when
                # valid_size * num_rows < batch_size and drop_last=True upstream.
                print("Validation loader is empty. Increase dataset size or lower batch_size.")
                return float("inf")
            valid_loss /= counter

        model.train()
        return valid_loss


def main(config):
    # Select the augmentation implementation requested by the YAML config.
    if config["aug"] == "node":
        from utils.dataset import MoleculeDatasetWrapper
    elif config["aug"] == "subgraph":
        from utils.dataset_subgraph import MoleculeDatasetWrapper
    elif config["aug"] == "mix":
        from utils.dataset_mix import MoleculeDatasetWrapper
    else:
        raise ValueError("Not defined molecule augmentation!")

    dataset = MoleculeDatasetWrapper(config["batch_size"], **config["dataset"])
    model_pretrain = PreTrain(dataset, config)
    model_pretrain.train()
    print(f"Training finished. Checkpoints saved in {model_pretrain.writer.log_dir}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the config file.")
    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    print(config)
    main(config)
