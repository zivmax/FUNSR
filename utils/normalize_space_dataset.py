import torch
import numpy as np
import os
from scipy.spatial import cKDTree
import trimesh
import open3d as o3d
from typing import Dict, Tuple, List


class NormalizeSpaceDataset(torch.utils.data.Dataset):
    def __init__(self, conf: Dict, dataname: str):
        super().__init__()
        self.device: torch.device = torch.device("cuda")  # Use CUDA if available
        self.conf: Dict = conf
        self.data_dir: str = conf.get_string("data_dir")
        self.np_data_name: str = dataname + ".pt"  # Preprocessed data file name
        self._load_or_process_data(dataname)

    def _load_or_process_data(self, dataname: str) -> None:
        """Loads preprocessed data or processes it if not available."""
        data_path: str = os.path.join(self.data_dir, self.np_data_name)
        if os.path.exists(data_path):
            print("Data existing. Loading data...")
            prep_data: Dict[str, torch.Tensor] = self._load_processed_data(data_path)
        else:
            print("Data not found. Processing data...")
            self.process_data(self.data_dir, dataname)
            prep_data: Dict[str, torch.Tensor] = self._load_processed_data(data_path)

        self.queries_nearest: torch.Tensor = prep_data[
            "queries_nearest"
        ]  # Points near the surface of the object.
        self.query_points: torch.Tensor = prep_data[
            "query_points"
        ]  # Query points used for sampling the space.
        self.points_gt: torch.Tensor = prep_data[
            "pointcloud"
        ]  # The original, full-resolution point cloud.

        self.sample_points_num: int = self.query_points.shape[0] - 1
        self._compute_bounding_box()
        print("NP Load data: End")

    def _load_processed_data(self, data_path: str) -> Dict[str, torch.Tensor]:
        """Loads preprocessed data from a .pt file."""
        return torch.load(data_path, map_location=self.device, weights_only=True)

    def _compute_bounding_box(self) -> None:
        """Computes and stores the bounding box of the downsampled point cloud."""
        self.object_bbox_min: torch.Tensor
        self.object_bbox_max: torch.Tensor
        self.object_bbox_min, _ = torch.min(self.queries_nearest, dim=0)
        self.object_bbox_min = self.object_bbox_min - 0.05  # Add padding
        self.object_bbox_max, _ = torch.max(self.queries_nearest, dim=0)
        self.object_bbox_max = self.object_bbox_max + 0.05  # Add padding
        print("Data bounding box:", self.object_bbox_min, self.object_bbox_max)

    def __len__(self) -> int:
        """Returns the total number of query points."""
        return self.query_points.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns a single data point (point, sample, point_gt)."""
        return self.queries_nearest[idx], self.query_points[idx], self.points_gt

    def np_train_data(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples a batch of data for training.

        Args:
            batch_size (int): The desired batch size.

        Returns:
            tuple: A tuple containing the sampled points, query points, and the original point cloud.
        """
        # The original logic seems to divide the query points into groups of 10.
        # It randomly picks one index from the first group (0-9) and 'batch_size' indices
        # from the remaining groups, then combines them.
        index_coarse: np.ndarray = np.random.choice(
            10, 1
        )  # Random index from the first 10
        index_fine: np.ndarray = np.random.choice(
            self.sample_points_num // 10, batch_size, replace=False
        )  # Random indices from the rest
        index: np.ndarray = index_fine * 10 + index_coarse  # Combine indices
        points: torch.Tensor = self.queries_nearest[index]
        sample: torch.Tensor = self.query_points[index]
        return points, sample, self.points_gt

    def process_data(self, data_dir: str, dataname: str) -> None:
        """Processes the raw point cloud data and saves it to a .pt file."""
        pointcloud: np.ndarray = self._load_raw_pointcloud(data_dir, dataname)
        pointcloud: np.ndarray = self._normalize_pointcloud(pointcloud)
        pointcloud: np.ndarray = self.FPS_sampling(pointcloud)  # Downsample using FPS
        pointcloud: torch.Tensor = torch.from_numpy(pointcloud).to(self.device).float()

        grid_f: torch.Tensor = self._generate_grid_points()
        query_points: torch.Tensor = self._generate_query_points(pointcloud)
        query_points: torch.Tensor = torch.cat(
            [query_points, grid_f]
        ).float()  # Combine query and grid points

        queries_nearest: torch.Tensor = self._find_nearest_neighbors(
            query_points, pointcloud
        )

        self._save_processed_data(
            data_dir, dataname, pointcloud, query_points, queries_nearest
        )

    def _load_raw_pointcloud(self, data_dir: str, dataname: str) -> np.ndarray:
        """Loads the raw point cloud data from either a .ply or .xyz file."""
        ply_path: str = os.path.join(data_dir, dataname + ".ply")
        xyz_path: str = os.path.join(data_dir, dataname + ".xyz")

        if os.path.exists(ply_path):
            pointcloud: np.ndarray = trimesh.load(ply_path).vertices
        elif os.path.exists(xyz_path):
            pointcloud: np.ndarray = np.loadtxt(
                xyz_path
            )  # Corrected: Use np.loadtxt for .xyz
        else:
            raise FileNotFoundError(
                "Only support .xyz or .ply data. Please adjust your data."
            )
        return np.asarray(pointcloud)

    def _normalize_pointcloud(self, pointcloud: np.ndarray) -> np.ndarray:
        """Normalizes the point cloud to center it and scale it to a unit sphere."""
        # Calculate the scale based on the maximum extent of the point cloud
        shape_scale: np.ndarray = np.max(
            np.ptp(pointcloud, axis=0)
        )  # ptp = peak-to-peak (max - min)
        shape_center: np.ndarray = np.mean(
            pointcloud, axis=0
        )  # Calculate the center of the point cloud
        pointcloud: np.ndarray = pointcloud - shape_center
        pointcloud: np.ndarray = pointcloud / shape_scale
        return pointcloud

    def _generate_grid_points(self, grid_samp: int = 30000) -> torch.Tensor:
        """Generates a set of grid points within the bounding box."""

        def gen_grid(start: float, end: float, num: int) -> np.ndarray:
            """Generates a 3D grid of points."""
            x: np.ndarray = np.linspace(start, end, num=num)
            y: np.ndarray = np.linspace(start, end, num=num)
            z: np.ndarray = np.linspace(start, end, num=num)
            g: List[np.ndarray] = np.meshgrid(x, y, z)
            positions: np.ndarray = np.vstack(list(map(np.ravel, g)))
            return positions.T  # Transpose to get (num_points, 3)

        dot5: np.ndarray = gen_grid(-0.5, 0.5, 70)  # Dense grid around the origin
        dot10: np.ndarray = gen_grid(-1.0, 1.0, 50)  # Less dense grid further away
        grid: np.ndarray = np.concatenate((dot5, dot10))
        grid: torch.Tensor = torch.from_numpy(grid).to(self.device).float()
        # Randomly sample a subset of the grid points
        grid_f: torch.Tensor = grid[torch.randperm(grid.shape[0])[:grid_samp]]
        return grid_f

    def _generate_query_points(
        self, pointcloud: torch.Tensor, query_per_point: int = 20
    ) -> torch.Tensor:
        """Generates query points around the downsampled point cloud."""
        # Use cKDTree for efficient nearest neighbor search
        ptree: cKDTree = cKDTree(pointcloud.detach().cpu().numpy())
        # Calculate standard deviations based on 51st nearest neighbor distances
        std: List[np.ndarray] = []
        for p in np.array_split(pointcloud.detach().cpu().numpy(), 100, axis=0):
            d = ptree.query(p, 51)  # Find distances to the 51 nearest neighbors
            std.append(d[0][:, -1])  # Use the distance to the 51st neighbor as std
        std: np.ndarray = np.concatenate(std)
        std: torch.Tensor = torch.from_numpy(std).to(self.device).float().unsqueeze(-1)

        query_points: List[torch.Tensor] = []
        for idx, p in enumerate(pointcloud):
            # Generate query points using a Gaussian distribution
            q_loc: torch.Tensor = (
                torch.normal(mean=0.0, std=std[idx].item(), size=(query_per_point, 3))
                .to(self.device)
                .float()
            )
            q: torch.Tensor = p + q_loc  # Add the offset to the downsampled point
            query_points.append(q)

        return torch.cat(query_points)

    def _find_nearest_neighbors(
        self,
        query_points: torch.Tensor,
        pointcloud: torch.Tensor,
        point_num: int = 1000,
    ) -> torch.Tensor:
        """Finds the nearest neighbor in the original point cloud for each query point."""
        # Process in chunks to avoid out-of-memory errors
        query_points_nn: torch.Tensor = torch.reshape(query_points, (-1, point_num, 3))
        queries_nearest: List[torch.Tensor] = []
        for j in range(query_points_nn.shape[0]):
            nearest_idx: np.ndarray = self.search_nearest_point(
                query_points_nn[j], pointcloud
            )
            nearest_points: torch.Tensor = pointcloud[nearest_idx]
            queries_nearest.append(nearest_points)
        return torch.cat(queries_nearest)

    def _save_processed_data(
        self,
        data_dir: str,
        dataname: str,
        pointcloud: torch.Tensor,
        query_points: torch.Tensor,
        queries_nearest: torch.Tensor,
    ) -> None:
        """Saves the processed data to a .pt file."""
        print("Saving files...")
        torch.save(
            {
                "pointcloud": pointcloud,
                "query_points": query_points,
                "queries_nearest": queries_nearest,
            },
            os.path.join(data_dir, dataname) + ".pt",
        )

    def FPS_sampling(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        Performs Farthest Point Sampling (FPS) on the point cloud.

        Args:
            point_cloud (np.ndarray): The input point cloud.
            data_dir (str): Directory to save intermediate files (if needed).
            dataname (str): Name of the data file.

        Returns:
            np.ndarray: The downsampled point cloud.
        """
        pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud.reshape(-1, 3))

        if len(pcd.points) > 60000:
            # If the point cloud is large, perform two-stage FPS
            pcd_down_1: o3d.geometry.PointCloud = pcd.farthest_point_down_sample(5000)
            pcd_down_2: o3d.geometry.PointCloud = pcd.farthest_point_down_sample(15000)
            pcdConcat: np.ndarray = np.concatenate(
                (np.asarray(pcd_down_1.points), np.asarray(pcd_down_2.points)), axis=0
            )
        else:
            # Otherwise, use the original point cloud
            pcdConcat: np.ndarray = np.asarray(pcd.points)

        print("number of points:", len(pcdConcat))
        return pcdConcat

    def search_nearest_point(
        self, point_batch: torch.Tensor, point_gt: torch.Tensor
    ) -> np.ndarray:
        """
        Finds the index of the nearest point in point_gt for each point in point_batch.

        Args:
            point_batch (torch.Tensor): Batch of query points (shape: [N, 3]).
            point_gt (torch.Tensor):  Ground truth point cloud (shape: [M, 3]).

        Returns:
            np.ndarray: Indices of the nearest neighbors in point_gt.
        """
        num_point_batch: int
        num_point_gt: int
        num_point_batch, num_point_gt = point_batch.shape[0], point_gt.shape[0]
        # Expand dimensions to enable broadcasting
        point_batch: torch.Tensor = point_batch.unsqueeze(1).expand(
            -1, num_point_gt, -1
        )
        point_gt: torch.Tensor = point_gt.unsqueeze(0).expand(num_point_batch, -1, -1)

        # Calculate squared Euclidean distances
        distances: torch.Tensor = torch.sum((point_batch - point_gt) ** 2, axis=-1)
        dis_idx: np.ndarray = torch.argmin(distances, axis=1).detach().cpu().numpy()
        return dis_idx
