import time
import torch
import torch.nn.functional as F
import argparse
import os
import math
import mcubes
import trimesh
import subprocess

from models.components.sdf_network import SDFNet
from models.components.discriminator import Discriminator
from models.datasets.normalize_space_dataset import NormalizeSpaceDataset
from utils.config import Config
from utils.logger import setup_logger, print_log
from tqdm import tqdm


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda")
        self.config = Config(args.conf, args.dataname)
        self.base_exp_dir = os.path.join(
            self.config.get_string("general.base_exp_dir"), args.dir
        )
        os.makedirs(self.base_exp_dir, exist_ok=True)

        self.logger = setup_logger(
            name="outs", log_file=os.path.join(self.base_exp_dir, "logger.log")
        )

        self.dataset = NormalizeSpaceDataset(
            self.config.get_config("dataset"), args.dataname
        )
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.config.get_int("train.batch_size"),
            shuffle=True,
        )

        # Networks
        self.sdf_network = SDFNet(**self.config.get_config("model.sdf_network")).to(
            self.device
        )
        self.discriminator = Discriminator(
            **self.config.get_config("model.discriminator")
        ).to(self.device)

        # Optimizers
        self.sdf_optimizer = torch.optim.Adam(
            self.sdf_network.parameters(),
            lr=self.config.get_float("train.learning_rate"),
        )
        self.dis_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.get_float("train.learning_rate"),
            betas=(0.9, 0.999),
        )

        self.iter_step = 0
        self.epoch = self.config.get_int("train.epoch")
        self.save_freq = self.config.get_int("train.save_freq")
        self.val_freq = self.config.get_int("train.val_freq")
        self.report_freq = self.config.get_int("train.report_freq")
        self.labmda_scc = self.config.get_float("train.labmda_scc")
        self.labmda_adl = self.config.get_float("train.labmda_adl")
        self.warm_up_end = self.config.get_float("train.warm_up_end")

        self.file_backup()

    def file_backup(self):
        os.makedirs(self.base_exp_dir, exist_ok=True)

        try:
            # Get git commit hash
            git_hash = (
                subprocess.check_output(["git", "rev-parse", "HEAD"])
                .decode("ascii")
                .strip()
            )

            # Save git hash to a file
            with open(os.path.join(self.base_exp_dir, "git_version.txt"), "w") as f:
                f.write(f"Git commit hash: {git_hash}\n")

        except Exception as e:
            print(f"Warning: Could not backup git version: {str(e)}")

    def update_learning_rate(self, iter_i):
        warm_up = self.warm_up_end
        max_iter = self.epoch * len(self.dataloader)
        init_lr = self.config.get_float("train.learning_rate")
        if iter_i < warm_up:
            lr = iter_i / warm_up
        else:
            lr = 0.5 * (
                math.cos((iter_i - warm_up) / (max_iter - warm_up) * math.pi) + 1
            )
        lr = lr * init_lr
        for g in self.sdf_optimizer.param_groups:
            g["lr"] = lr

    def get_discriminator_loss_single(self, pred, label=True):
        if label:
            return torch.mean((pred - 1) ** 2)
        else:
            return torch.mean((pred) ** 2)

    def get_funsr_loss(self, pred_fake):
        return torch.mean((pred_fake - 1) ** 2)

    def train(self):
        total_loss = torch.tensor(0.0)

        for epoch in range(self.epoch):
            epoch_loss = 0.0
            epoch_steps = 0
            progress_bar = tqdm(
                self.dataloader,
                desc=f"Epoch {epoch+1}/{self.epoch}",
                leave=True,
                dynamic_ncols=True,
            )
            for points, samples, point_gt in progress_bar:
                self.update_learning_rate(self.iter_step)
                points, samples, point_gt = (
                    points.to(self.device),
                    samples.to(self.device),
                    point_gt.to(self.device),
                )

                # Train FUNSR Network
                self.sdf_optimizer.zero_grad()
                samples.requires_grad = True
                gradients_sample = self.sdf_network.gradient(samples).squeeze()
                sdf_sample = self.sdf_network.sdf(samples)
                grad_norm = F.normalize(gradients_sample, dim=1)
                sample_moved = samples - grad_norm * sdf_sample

                loss_sdf = torch.linalg.norm(
                    (points - sample_moved), ord=2, dim=-1
                ).mean()
                SCC = F.normalize(sample_moved - points, dim=1)
                loss_SCC = (1.0 - F.cosine_similarity(grad_norm, SCC, dim=1)).mean()
                G_loss = loss_sdf + loss_SCC * self.labmda_scc

                # Train Discriminator
                self.dis_optimizer.zero_grad()
                d_fake_output = self.discriminator.sdf(sdf_sample.detach())
                d_fake_loss = self.get_discriminator_loss_single(
                    d_fake_output, label=False
                )

                real_sdf = torch.zeros(points.size(0), 1).to(self.device)
                d_real_output = self.discriminator.sdf(real_sdf)
                d_real_loss = self.get_discriminator_loss_single(
                    d_real_output, label=True
                )
                dis_loss = d_real_loss + d_fake_loss
                dis_loss.backward()
                self.dis_optimizer.step()

                # Total Loss
                d_fake_output = self.discriminator.sdf(sdf_sample)
                gan_loss = self.get_funsr_loss(d_fake_output)
                total_loss = gan_loss * self.labmda_adl + G_loss
                total_loss.backward()
                self.sdf_optimizer.step()

                epoch_loss += total_loss.item()
                epoch_steps += 1

                # Logging and Validation
                self.iter_step += 1

                progress_bar.set_postfix(
                    {
                        "LR": f"{self.sdf_optimizer.param_groups[0]['lr']:.6f}",
                        "AvgL": (
                            f"{epoch_loss / epoch_steps:.6f}"
                            if epoch_steps > 0
                            else "0.000000"
                        ),
                    }
                )

            if epoch % self.val_freq == 0:
                self.validate_mesh(
                    resolution=256,
                    threshold=self.args.mcubes_threshold,
                    point_gt=point_gt,
                    iter_step=self.iter_step,
                )

            if epoch % self.save_freq == 0:
                self.save_checkpoint()

    def validate_mesh(self, resolution, threshold, point_gt, iter_step):
        output_dir = os.path.join(self.base_exp_dir, "outputs")
        os.makedirs(output_dir, exist_ok=True)
        mesh = self.extract_geometry(
            resolution=resolution,
            threshold=threshold,
            query_func=lambda pts: -self.sdf_network.sdf(pts),
        )
        mesh.export(os.path.join(output_dir, f"{iter_step:08d}_{threshold}.ply"))

    def extract_geometry(self, resolution, threshold, query_func):
        print_log(f"Creating mesh with threshold: {threshold}", logger=self.logger)
        u = self.extract_fields(resolution, query_func).numpy()
        vertices, triangles = mcubes.marching_cubes(u, threshold)

        voxel_origin = [-1, -1, -1]
        voxel_size = 2.0 / (resolution - 1)

        vertices[:, 0] = vertices[:, 0] * voxel_size + voxel_origin[0]
        vertices[:, 1] = vertices[:, 1] * voxel_size + voxel_origin[0]
        vertices[:, 2] = vertices[:, 2] * voxel_size + voxel_origin[0]

        mesh = trimesh.Trimesh(vertices, triangles)

        return mesh

    def create_cube(self, N):

        overall_index = torch.arange(0, N**3, 1, out=torch.LongTensor())
        samples = torch.zeros(N**3, 4)

        # the voxel_origin is the (bottom, left, down) corner, not the middle
        voxel_origin = [-1, -1, -1]
        voxel_size = 2.0 / (N - 1)

        # transform first 3 columns
        # to be the x, y, z index
        samples[:, 2] = overall_index % N
        samples[:, 1] = (overall_index.long().float() / N) % N
        samples[:, 0] = ((overall_index.long().float() / N) / N) % N

        # transform first 3 columns
        # to be the x, y, z coordinate
        samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
        samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
        samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

        samples.requires_grad = False

        return samples

    def extract_fields(self, resolution, query_func):
        N = resolution
        max_batch = 1000000
        # the voxel_origin is the (bottom, left, down) corner, not the middle
        cube = self.create_cube(resolution).cuda()
        cube_points = cube.shape[0]

        with torch.no_grad():
            head = 0
            while head < cube_points:

                query = cube[head : min(head + max_batch, cube_points), 0:3]

                # inference defined in forward function per pytorch lightning convention
                pred_sdf = query_func(query)

                cube[head : min(head + max_batch, cube_points), 3] = pred_sdf.squeeze()

                head += max_batch

        # for occupancy instead of SDF, subtract 0.5 so the surface boundary becomes 0
        sdf_values = cube[:, 3]
        sdf_values = sdf_values.reshape(N, N, N).detach().cpu()

        return sdf_values

    def save_checkpoint(self):
        checkpoint = {
            "sdf_network": self.sdf_network.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "sdf_optimizer": self.sdf_optimizer.state_dict(),
            "dis_optimizer": self.dis_optimizer.state_dict(),
            "iter_step": self.iter_step,
        }
        os.makedirs(os.path.join(self.base_exp_dir, "checkpoints"), exist_ok=True)
        torch.save(
            checkpoint,
            os.path.join(
                self.base_exp_dir,
                "checkpoints",
                f"ckpt_{self.iter_step:06d}.pth",
            ),
        )

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(
            os.path.join(self.base_exp_dir, "checkpoints", checkpoint_name),
            map_location=self.device,
            weights_only=True,
        )
        self.sdf_network.load_state_dict(checkpoint["sdf_network"])
        self.discriminator.load_state_dict(checkpoint["discriminator"])
        self.sdf_optimizer.load_state_dict(checkpoint["sdf_optimizer"])
        self.dis_optimizer.load_state_dict(checkpoint["dis_optimizer"])
        self.iter_step = checkpoint["iter_step"]
        print_log(
            f"Loaded checkpoint {checkpoint_name} at iter {self.iter_step}", self.logger
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default="./confs/conf.conf")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--mcubes_threshold", type=float, default=0.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dir", type=str, default="exp")
    parser.add_argument("--dataname", type=str, default="case000070.nii_ds")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    trainer = Trainer(args)

    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    if args.mode == "train":
        trainer.train()
