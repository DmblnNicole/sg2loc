"""
Adapted from SceneGraphLoc: https://github.com/y9miao/VLSG.
"""

import numpy as np


def load_inseg(pth_ply):
    # trimesh stays a local import, the refinement environment does not ship it
    import trimesh

    cloud_pd = trimesh.load(pth_ply, process=False)
    points_pd = cloud_pd.vertices
    segments_pd = cloud_pd.metadata["_ply_raw"]["vertex"]["data"]["label"].flatten()

    return cloud_pd, points_pd, segments_pd


def pcl_farthest_sample(point, npoint, return_idxs=False):
    """Sample npoint points from the cloud by farthest-point sampling."""
    N, D = point.shape
    if N < npoint:
        indices = np.random.choice(point.shape[0], npoint)
        point = point[indices]
        return point

    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]

    if return_idxs:
        return point, centroids.astype(np.int32)
    return point


def load_plydata_npy(file_path, obj_ids=None, return_ply_data=False):
    ply_data = np.load(file_path)
    points = np.stack([ply_data["x"], ply_data["y"], ply_data["z"]]).transpose((1, 0))

    if obj_ids is not None:
        if type(obj_ids) == np.ndarray:  # noqa: E721
            obj_ids_pc = ply_data["objectId"]
            obj_ids_pc_mask = np.isin(obj_ids_pc, obj_ids)
            points = points[np.where(obj_ids_pc_mask == True)[0]]  # noqa: E712
        else:
            obj_ids_pc = ply_data["objectId"]
            points = points[np.where(obj_ids_pc == obj_ids)[0]]

    if return_ply_data:
        return points, ply_data
    else:
        return points
