import torch
import numpy as np
import os
from scipy.spatial import cKDTree
import trimesh
import open3d as o3d


class NormalizeSpaceDataset(torch.utils.data.Dataset):
    def __init__(self, conf, dataname):
        super().__init__()
        self.device = torch.device("cuda")  # Use CUDA if available
        self.conf = conf
        self.data_dir = conf.get_string("data_dir")
        self.np_data_name = dataname + ".pt"  # Preprocessed data file name
        self._load_or_process_data(dataname)

    def _load_or_process_data(self, dataname):
        """Loads preprocessed data or processes it if not available."""
        data_path = os.path.join(self.data_dir, self.np_data_name)
        if os.path.exists(data_path):
            print("Data existing. Loading data...")
            prep_data = self._load_processed_data(data_path)
        else:
            print("Data not found. Processing data...")
            self.process_data(self.data_dir, dataname)
            prep_data = self._load_processed_data(data_path)

        self.points = prep_data["sample_near"]  # Points near the surface of the object.
        self.samples = prep_data[
            "query_points"
        ]  # Query points used for sampling the space.
        self.points_gt = prep_data[
            "pointcloud"
        ]  # The original, full-resolution point cloud.

        self.sample_points_num = self.samples.shape[0] - 1
        self._compute_bounding_box()
        print("NP Load data: End")

    def _load_processed_data(self, data_path):
        """Loads preprocessed data from a .pt file."""
        return torch.load(data_path, map_location=self.device, weights_only=True)

    def _compute_bounding_box(self):
        """Computes and stores the bounding box of the downsampled point cloud."""
        self.object_bbox_min, _ = torch.min(self.points, dim=0)
        self.object_bbox_min = self.object_bbox_min - 0.05  # Add padding
        self.object_bbox_max, _ = torch.max(self.points, dim=0)
        self.object_bbox_max = self.object_bbox_max + 0.05  # Add padding
        print("Data bounding box:", self.object_bbox_min, self.object_bbox_max)

    def __len__(self):
        """Returns the total number of query points."""
        return self.samples.shape[0]

    def __getitem__(self, idx):
        """Returns a single data point (point, sample, point_gt)."""
        return self.points[idx], self.samples[idx], self.points_gt

    def np_train_data(self, batch_size):
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
        index_coarse = np.random.choice(10, 1)  # Random index from the first 10
        index_fine = np.random.choice(
            self.sample_points_num // 10, batch_size, replace=False
        )  # Random indices from the rest
        index = index_fine * 10 + index_coarse  # Combine indices
        points = self.points[index]
        sample = self.samples[index]
        return points, sample, self.points_gt

    def process_data(self, data_dir, dataname):
        """Processes the raw point cloud data and saves it to a .pt file."""
        pointcloud = self._load_raw_pointcloud(data_dir, dataname)
        pointcloud = self._normalize_pointcloud(pointcloud)
        pointcloud = self.FPS_sampling(pointcloud)  # Downsample using FPS
        pointcloud = torch.from_numpy(pointcloud).to(self.device).float()

        grid_f = self._generate_grid_points()
        query_points = self._generate_query_points(pointcloud)
        query_points = torch.cat(
            [query_points, grid_f]
        ).float()  # Combine query and grid points

        sample_near = self._find_nearest_neighbors(query_points, pointcloud)

        self._save_processed_data(
            data_dir, dataname, pointcloud, query_points, sample_near
        )

    def _load_raw_pointcloud(self, data_dir, dataname):
        """Loads the raw point cloud data from either a .ply or .xyz file."""
        ply_path = os.path.join(data_dir, dataname + ".ply")
        xyz_path = os.path.join(data_dir, dataname + ".xyz")

        if os.path.exists(ply_path):
            pointcloud = trimesh.load(ply_path).vertices
        elif os.path.exists(xyz_path):
            pointcloud = np.loadtxt(xyz_path)  # Corrected: Use np.loadtxt for .xyz
        else:
            raise FileNotFoundError(
                "Only support .xyz or .ply data. Please adjust your data."
            )
        return np.asarray(pointcloud)

    def _normalize_pointcloud(self, pointcloud):
        """Normalizes the point cloud to center it and scale it to a unit sphere."""
        # Calculate the scale based on the maximum extent of the point cloud
        shape_scale = np.max(
            np.ptp(pointcloud, axis=0)
        )  # ptp = peak-to-peak (max - min)
        shape_center = np.mean(
            pointcloud, axis=0
        )  # Calculate the center of the point cloud
        pointcloud = pointcloud - shape_center
        pointcloud = pointcloud / shape_scale
        return pointcloud

    def _generate_grid_points(self, grid_samp=30000):
        """Generates a set of grid points within the bounding box."""

        def gen_grid(start, end, num):
            """Generates a 3D grid of points."""
            x = np.linspace(start, end, num=num)
            y = np.linspace(start, end, num=num)
            z = np.linspace(start, end, num=num)
            g = np.meshgrid(x, y, z)
            positions = np.vstack(list(map(np.ravel, g)))
            return positions.T  # Transpose to get (num_points, 3)

        dot5 = gen_grid(-0.5, 0.5, 70)  # Dense grid around the origin
        dot10 = gen_grid(-1.0, 1.0, 50)  # Less dense grid further away
        grid = np.concatenate((dot5, dot10))
        grid = torch.from_numpy(grid).to(self.device).float()
        # Randomly sample a subset of the grid points
        grid_f = grid[torch.randperm(grid.shape[0])[:grid_samp]]
        return grid_f

    def _generate_query_points(self, pointcloud, query_per_point=20):
        """Generates query points around the downsampled point cloud."""
        # Use cKDTree for efficient nearest neighbor search
        ptree = cKDTree(pointcloud.detach().cpu().numpy())
        # Calculate standard deviations based on 51st nearest neighbor distances
        std = []
        for p in np.array_split(pointcloud.detach().cpu().numpy(), 100, axis=0):
            d = ptree.query(p, 51)  # Find distances to the 51 nearest neighbors
            std.append(d[0][:, -1])  # Use the distance to the 51st neighbor as std
        std = np.concatenate(std)
        std = torch.from_numpy(std).to(self.device).float().unsqueeze(-1)

        query_points = []
        for idx, p in enumerate(pointcloud):
            # Generate query points using a Gaussian distribution
            q_loc = (
                torch.normal(mean=0.0, std=std[idx].item(), size=(query_per_point, 3))
                .to(self.device)
                .float()
            )
            q = p + q_loc  # Add the offset to the downsampled point
            query_points.append(q)

        return torch.cat(query_points)

    def _find_nearest_neighbors(self, query_points, pointcloud, point_num=1000):
        """Finds the nearest neighbor in the original point cloud for each query point."""
        # Process in chunks to avoid out-of-memory errors
        query_points_nn = torch.reshape(query_points, (-1, point_num, 3))
        sample_near = []
        for j in range(query_points_nn.shape[0]):
            nearest_idx = self.search_nearest_point(query_points_nn[j], pointcloud)
            nearest_points = pointcloud[nearest_idx]
            sample_near.append(nearest_points)
        return torch.cat(sample_near)

    def _save_processed_data(
        self, data_dir, dataname, pointcloud, query_points, sample_near
    ):
        """Saves the processed data to a .pt file."""
        print("Saving files...")
        torch.save(
            {
                "pointcloud": pointcloud,
                "query_points": query_points,
                "sample_near": sample_near,
            },
            os.path.join(data_dir, dataname) + ".pt",
        )

    def FPS_sampling(self, point_cloud):
        """
        Performs Farthest Point Sampling (FPS) on the point cloud.

        Args:
            point_cloud (np.ndarray): The input point cloud.
            data_dir (str): Directory to save intermediate files (if needed).
            dataname (str): Name of the data file.

        Returns:
            np.ndarray: The downsampled point cloud.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud.reshape(-1, 3))

        if len(pcd.points) > 60000:
            # If the point cloud is large, perform two-stage FPS
            pcd_down_1 = pcd.farthest_point_down_sample(5000)
            pcd_down_2 = pcd.farthest_point_down_sample(15000)
            pcdConcat = np.concatenate(
                (np.asarray(pcd_down_1.points), np.asarray(pcd_down_2.points)), axis=0
            )
        else:
            # Otherwise, use the original point cloud
            pcdConcat = np.asarray(pcd.points)

        print("number of points:", len(pcdConcat))
        return pcdConcat

    def search_nearest_point(self, point_batch, point_gt):
        """
        Finds the index of the nearest point in point_gt for each point in point_batch.

        Args:
            point_batch (torch.Tensor): Batch of query points (shape: [N, 3]).
            point_gt (torch.Tensor):  Ground truth point cloud (shape: [M, 3]).

        Returns:
            np.ndarray: Indices of the nearest neighbors in point_gt.
        """
        num_point_batch, num_point_gt = point_batch.shape[0], point_gt.shape[0]
        # Expand dimensions to enable broadcasting
        point_batch = point_batch.unsqueeze(1).expand(-1, num_point_gt, -1)
        point_gt = point_gt.unsqueeze(0).expand(num_point_batch, -1, -1)

        # Calculate squared Euclidean distances
        distances = torch.sum((point_batch - point_gt) ** 2, axis=-1)
        dis_idx = torch.argmin(distances, axis=1).detach().cpu().numpy()
        return dis_idx
