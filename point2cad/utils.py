import matplotlib
import numpy as np
import open3d as o3d
import os
import pathlib
import random
import scipy
import torch
import trimesh
from PIL import Image
from contextlib import contextmanager
from geomdl.tessellate import make_triangle_mesh

EPS = np.finfo(np.float32).eps


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def regular_parameterization(grid_u, grid_v):
    nx, ny = (grid_u, grid_v)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xv, yv = np.meshgrid(x, y)
    xv = np.expand_dims(xv.transpose().flatten(), 1)
    yv = np.expand_dims(yv.transpose().flatten(), 1)
    parameters = np.concatenate([xv, yv], 1)
    return parameters


def get_rotation_matrix(theta):
    R = np.array(
        [
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    return R


def normalize_points(points, anisotropic=False):
    points = points - np.mean(points, 0, keepdims=True)
    # noise = normals * np.clip(np.random.randn(points.shape[0], 1) * 0.01, a_min=-0.01, a_max=0.01)
    # points = points + noise.astype(np.float32)

    S, U = pca_numpy(points)
    smallest_ev = U[:, np.argmin(S)]
    R = rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))
    # rotate input points such that the minor principal
    # axis aligns with x axis.
    points = (R @ points.T).T
    std = np.max(points, 0) - np.min(points, 0)
    if anisotropic:
        points = points / (std.reshape((1, 3)) + EPS)
    else:
        points = points / (np.max(std) + EPS)
    return points.astype(np.float32)


def pca_numpy(X):
    S, U = np.linalg.eig(X.T @ X)
    return S, U


def rotation_matrix_a_to_b(A, B):
    """
    Finds rotation matrix from vector A in 3d to vector B
    in 3d.
    B = R @ A
    """
    cos = np.dot(A, B)
    sin = np.linalg.norm(np.cross(B, A))
    u = A
    v = B - np.dot(A, B) * A
    v = v / (np.linalg.norm(v) + EPS)
    w = np.cross(B, A)
    w = w / (np.linalg.norm(w) + EPS)
    F = np.stack([u, v, w], 1)
    G = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
    # B = R @ A
    try:
        R = F @ G @ np.linalg.inv(F)
    except:
        R = np.eye(3, dtype=np.float32)
    return R


def continuous_labels(labels_):
    new_labels = np.zeros_like(labels_)
    for index, value in enumerate(np.sort(np.unique(labels_))):
        new_labels[labels_ == value] = index
    return new_labels


def tessalate_points(points, size_u, size_v, mask=None):
    points = points.reshape((size_u * size_v, 3))
    points = [list(points[i, :]) for i in range(points.shape[0])]
    vertex, triangle = make_triangle_mesh(points, size_u, size_v, mask=mask)

    triangle = [t.data for t in triangle]
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(vertex))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangle))
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


def make_colormap_optimal(num_colors=64):
    num_colors_x = int(np.sqrt(num_colors))
    assert num_colors_x**2 == num_colors
    x_range = np.arange(0, num_colors_x) / (num_colors_x - 1)
    c_y = np.repeat(x_range.reshape(-1, 1), num_colors_x, axis=1)
    c_x = np.repeat(x_range.reshape(1, -1), num_colors_x, axis=0)
    c_yx = np.stack((c_y, c_x), axis=-1).reshape(-1, 2)
    c_dist = scipy.spatial.distance.cdist(c_yx, c_yx)
    pos = (num_colors_x - 1) * num_colors_x
    palette = [pos]
    ranking_remaining = c_dist[pos]
    for i in range(1, num_colors):
        # greedily choose the most distant position
        for p in palette:
            ranking_remaining[p] = 0
        pos = np.argmax(ranking_remaining)
        palette.append(pos)
        ranking_remaining += c_dist[pos]
    path_colormap = str(
        pathlib.Path(__file__).parent.parent.resolve()
        / "dependencies"
        / "perceptual_colormap"
        / "colormap2d.png"
    )
    palette_img = np.array(Image.open(path_colormap))
    palette = [
        palette_img[
            int((palette_img.shape[0] - 1) * (c // num_colors_x) / (num_colors_x - 1)),
            int((palette_img.shape[1] - 1) * (c % num_colors_x) / (num_colors_x - 1)),
        ][:3]
        for c in palette
    ]
    return palette


@contextmanager
def suppress_output_fd():
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    with open(os.devnull, "w") as fnull:
        devnull_fd = fnull.fileno()
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        try:
            yield
        finally:
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)


def get_rng(device, seed=None, seed_increment=0):
    if seed is None:
        if device == "cpu":
            rng = torch.random.default_generator
        else:
            if isinstance(device, str):
                device = torch.device(device)
            elif isinstance(device, int):
                device = torch.device("cuda", device)
            device_idx = device.index
            if device_idx is None:
                device_idx = torch.cuda.current_device()
            rng = torch.cuda.default_generators[device_idx]
    else:
        rng = torch.Generator(device)
        rng.manual_seed(seed + seed_increment)
    return rng


def sample_inr_points(fit_out, mesh_dim=20, uv_margin=0.2, return_uv=False):
    uv_bb_sz = fit_out["uv_bb_max"] - fit_out["uv_bb_min"]
    uv_bb_margin = uv_bb_sz * uv_margin
    uv_min = fit_out["uv_bb_min"] - uv_bb_margin
    uv_max = fit_out["uv_bb_max"] + uv_bb_margin
    if fit_out["is_u_closed"]:
        uv_min[0] = max(uv_min[0], -1)
        uv_max[0] = min(uv_max[0], 1)
    if fit_out["is_v_closed"]:
        uv_min[1] = max(uv_min[1], -1)
        uv_max[1] = min(uv_max[1], 1)

    model = fit_out["model"]

    device = next(model.parameters()).device
    u, v = torch.meshgrid(
        torch.linspace(uv_min[0].item(), uv_max[0].item(), mesh_dim, device=device),
        torch.linspace(uv_min[1].item(), uv_max[1].item(), mesh_dim, device=device),
        indexing="xy",
    )
    uv = torch.stack((u, v), dim=2)  # M x M x 2
    model.eval()
    with torch.no_grad():
        points = model.decoder(uv.reshape(-1, 2))
    points3d_scale = fit_out["points3d_scale"]
    points3d_offset = fit_out["points3d_offset"]
    if not torch.is_tensor(points3d_scale):
        points3d_scale = torch.tensor(points3d_scale, device=device)
    if not torch.is_tensor(points3d_offset):
        points3d_offset = torch.tensor(points3d_offset, device=device)
    points = points * points3d_scale + points3d_offset
    if return_uv:
        return points, uv
    return points


def sample_inr_mesh(fit_out, mesh_dim=20, uv_margin=0.2):
    points, uv = sample_inr_points(
        fit_out, mesh_dim=mesh_dim, uv_margin=uv_margin, return_uv=True
    )
    faces = []
    vertex_colors = []
    path_colormap = str(
        pathlib.Path(__file__).parent.parent.resolve()
        / "dependencies"
        / "perceptual_colormap"
        / "colormap2d.png"
    )
    colormap = Image.open(path_colormap)
    cm_w, cm_h = colormap.size
    for i in range(mesh_dim):
        for j in range(mesh_dim):
            if i < mesh_dim - 1 and j < mesh_dim - 1:
                faces.append(
                    [
                        i * mesh_dim + j,
                        (i + 1) * mesh_dim + j,
                        i * mesh_dim + j + 1,
                    ]
                )
                faces.append(
                    [
                        i * mesh_dim + j + 1,
                        (i + 1) * mesh_dim + j,
                        (i + 1) * mesh_dim + j + 1,
                    ]
                )
            cm_i = i * (cm_h - 1) / (mesh_dim - 1)
            cm_j = j * (cm_w - 1) / (mesh_dim - 1)
            vertex_colors.append(colormap.getpixel((cm_j, cm_i)))
    faces = torch.tensor(faces, device=points.device)
    vertex_colors = torch.tensor(vertex_colors, dtype=torch.uint8, device=points.device)
    out = trimesh.Trimesh(
        points.numpy(),
        faces.numpy(),
        vertex_colors=vertex_colors.numpy(),
    )
    return out


def guard_exp(x, max_value=75, min_value=-75):
    x = torch.clamp(x, max=max_value, min=min_value)
    return torch.exp(x)


def guard_sqrt(x, minimum=1e-5):
    x = torch.clamp(x, min=minimum)
    return torch.sqrt(x)
