import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro
from gsplat.project_gaussians import _ProjectGaussians
from gsplat.rasterize import _RasterizeGaussians
from PIL import Image
from torch import Tensor, optim
import cv2


class SimpleTrainer:
    """Trains random gaussians to fit an image."""

    def __init__(self, gt_image: Tensor, num_points: int = 200, random=True):
        self.device = torch.device("cuda:0")
        self.gt_image = gt_image.to(device=self.device)
        self.num_points = num_points

        BLOCK_X, BLOCK_Y = 16, 16
        fov_x = math.pi / 2.0
        self.H, self.W = gt_image.shape[0], gt_image.shape[1]
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)

        self.tile_bounds = (
            (self.W + BLOCK_X - 1) // BLOCK_X,
            (self.H + BLOCK_Y - 1) // BLOCK_Y,
            1,
        )
        self.img_size = torch.tensor([self.W, self.H, 1], device=self.device)
        self.block = torch.tensor([BLOCK_X, BLOCK_Y, 1], device=self.device)

        self._init_gaussians(random=random)

    def _init_gaussians(self, random=True):
        """Random gaussians"""
        # print(f"Random={random}")
        if random:
            n = self.num_points * self.num_points
            self.means = 2.0 * (torch.rand(n, 3, device=self.device) - 0.5)
            self.scales = torch.rand(n, 3, device=self.device) # self.num_points
            self.rgbs = torch.rand(n, 3, device=self.device) # self.num_points
        else:
            pos_x, pos_y = torch.meshgrid(torch.arange(-1.0, 1.0, 1.0 / self.num_points), \
                                      torch.arange(-1.0, 1.0, 1.0 / self.num_points), indexing="ij")
            self.means = torch.stack([pos_x.reshape(-1), pos_y.reshape(-1), torch.zeros_like(pos_x).reshape(-1)], axis=-1).to(device=self.device)
            n = self.means.shape[0] # self.num_points * self.num_points
            self.scales = 1.0 / self.num_points * torch.ones((n, 3), device=self.device) # torch.rand(n, 3, device=self.device) # self.num_points
            self.rgbs = 0.5 * torch.ones((n, 3), device=self.device) # torch.rand(n, 3, device=self.device) # self.num_points

        u = torch.rand(n, 1, device=self.device) # rand
        v = torch.rand(n, 1, device=self.device)
        w = torch.rand(n, 1, device=self.device)
        self.quats = torch.cat(
            [
                torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
            ],
            -1,
        )
        if not random:
            _u, _v, _w = 1, 1, 1
            for i, x in enumerate([_u * _v, _u * (1-_v), (1-_u) * _w, (1-_u) * (1-_w)]):
                self.quats[:, i] = x
            # self.quats = torch.ones((n, 4), device=self.device) # ignore second order color
        if random:
            self.opacities = torch.rand((n, 1), device=self.device)  # self.num_points
        else:
            self.opacities = torch.ones((n, 1), device=self.device)#self.num_points

        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],#0.0, 0.0, 1.0, 8.0
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )
        self.background = torch.zeros(3, device=self.device)

        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.quats.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = True
        self.viewmat.requires_grad = False

    def train(self, iterations: int = 1000, lr: float = 0.01, save_imgs: bool = False):
        optimizer = optim.Adam(
            [self.rgbs, self.means, self.scales, self.opacities, self.quats], lr
        )
        mse_loss = torch.nn.MSELoss()
        frames = []
        times = [0] * 3  # project, rasterize, backward
        for iter in range(iterations):
            start = time.time()
            xys, depths, radii, conics, num_tiles_hit, cov3d = _ProjectGaussians.apply(
                self.means,
                self.scales,
                1,
                self.quats,
                self.viewmat,
                self.viewmat,
                self.focal,
                self.focal,
                self.W / 2,
                self.H / 2,
                self.H,
                self.W,
                self.tile_bounds,
            )
            torch.cuda.synchronize()
            times[0] += time.time() - start
            start = time.time()
            out_img = _RasterizeGaussians.apply(
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,
                torch.sigmoid(self.rgbs),
                torch.sigmoid(self.opacities),
                self.H,
                self.W,
                self.background,
            )
            torch.cuda.synchronize()
            times[1] += time.time() - start
            loss = mse_loss(out_img, self.gt_image)
            optimizer.zero_grad()
            start = time.time()
            loss.backward()
            torch.cuda.synchronize()
            times[2] += time.time() - start
            optimizer.step()
            print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")
            frame = (out_img.detach().cpu().numpy() * 255).astype(np.uint8)

            if save_imgs and iter % 5 == 0:
                frames.append(frame)
            if iter == 0:
                cv2.imwrite("test.jpg", frame * 255)
                # exit(0)


        print("Resulting params: ")
        print(self.means)
        print(self.scales)
        if save_imgs:
            # save them as a gif with PIL
            frames = [Image.fromarray(frame) for frame in frames]
            out_dir = os.path.join(os.getcwd(), "renders")
            os.makedirs(out_dir, exist_ok=True)
            frames[0].save(
                f"{out_dir}/training.gif",
                save_all=True, append_images=frames[1:],
                optimize=False, duration=5, loop=0,
            )
        print(f"Total(s):\nProject: {times[0]:.3f}, Rasterize: {times[1]:.3f}, Backward: {times[2]:.3f}")
        print(f"Per step(s):\nProject: {times[0]/iterations:.5f}, Rasterize: {times[1]/iterations:.5f}, Backward: {times[2]/iterations:.5f}")


def image_path_to_tensor(image_path: Path):
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor


def main(
    height: int = 256,
    width: int = 256,
    num_points: int = 100,#100000,
    save_imgs: bool = True,
    img_path: Optional[Path] = None,
    iterations: int = 1000,
    lr: float = 0.01,
) -> None:
    if img_path:
        gt_image = image_path_to_tensor(img_path)
    else:
        gt_image = torch.ones((height, width, 3)) * 1.0
        # make top left and bottom right red, blue
        gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0])
        gt_image[height // 2 :, width // 2 :, :] = torch.tensor([0.0, 0.0, 1.0])

    trainer = SimpleTrainer(gt_image=gt_image, num_points=num_points, random=True)
    trainer.train(iterations=iterations, lr=lr, save_imgs=save_imgs)


if __name__ == "__main__":
    tyro.cli(main)