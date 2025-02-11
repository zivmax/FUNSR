import time
import torch
import torch.nn.functional as F
import argparse
import os
import math
import mcubes
import trimesh
import subprocess

from models.sdf import SDFNet
from models.discriminator import Discriminator
from utils.normalize_space_dataset import NormalizeSpaceDataset
from utils.config import Config
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
            for queries_nearest, queries, _ in progress_bar:
                self.update_learning_rate(self.iter_step)
                queries_nearest, queries = (
                    queries_nearest.to(self.device),
                    queries.to(self.device),
                )

                # Train SDF Network
                self.sdf_optimizer.zero_grad()
                queries.requires_grad = True
                gradients_query = self.sdf_network.gradient(queries).squeeze()
                sds_pred = self.sdf_network.sdf(queries)
                grads_norm = F.normalize(gradients_query, dim=1)
                sds_pred_error = queries - grads_norm * sds_pred

                loss_sdf = torch.linalg.norm(
                    (queries_nearest - sds_pred_error), ord=2, dim=-1
                ).mean()

                SCC = F.normalize(sds_pred_error - queries_nearest, dim=1)
                loss_SCC = (1.0 - F.cosine_similarity(grads_norm, SCC, dim=1)).mean()
                G_loss = loss_sdf + loss_SCC * self.labmda_scc

                # Train Discriminator
                self.dis_optimizer.zero_grad()
                d_fake_output = self.discriminator.sdf(sds_pred.detach())
                d_fake_loss = torch.mean((d_fake_output) ** 2)

                sds_real = torch.zeros(queries_nearest.size(0), 1).to(self.device)
                d_real_output = self.discriminator.sdf(sds_real)
                d_real_loss = torch.mean((d_real_output - 1) ** 2)
                dis_loss = d_real_loss + d_fake_loss
                dis_loss.backward()
                self.dis_optimizer.step()

                # Total Loss
                d_fake_output = self.discriminator.sdf(sds_pred)
                GAN_loss = torch.mean((d_fake_output - 1) ** 2)
                total_loss = GAN_loss * self.labmda_adl + G_loss
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
                    iter_step=self.iter_step,
                )

            if epoch % self.save_freq == 0:
                self.save_checkpoint()

    def validate_mesh(self, resolution, threshold, iter_step):
        output_dir = os.path.join(self.base_exp_dir, "outputs")
        os.makedirs(output_dir, exist_ok=True)
        mesh = self.extract_geometry(
            resolution=resolution,
            threshold=threshold,
            query_func=lambda pts: -self.sdf_network.sdf(pts),
        )
        mesh.export(os.path.join(output_dir, f"{iter_step:08d}_{threshold}.ply"))

    def extract_geometry(self, resolution, threshold, query_func):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default="confs/conf.conf")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--mcubes_threshold", type=float, default=0.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dir", type=str, default="exp")
    parser.add_argument("--dataname", type=str, default="case000070.nii_ds")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    trainer = Trainer(args)

    match args.mode:
        case "train":
            trainer.train()

        case "eval":
            if args.checkpoint:
                trainer.load_checkpoint(args.checkpoint)
            trainer.validate_mesh(
                resolution=256,
                threshold=args.mcubes_threshold,
                iter_step=0,
            )
        case _:
            raise ValueError(f"Invalid mode: {args.mode}")
