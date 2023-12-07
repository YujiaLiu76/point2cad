import argparse
import numpy as np
from src.PointNet import PrimitivesEmbeddingDGCNGn
from src.fitting_utils import pca_numpy

# from src.segment_loss import EmbeddingLoss
from src.mean_shift import MeanShift
from src.segment_utils import rotation_matrix_a_to_b


def guard_mean_shift(embedding, quantile, iterations, kernel_type="gaussian"):
    """
    Sometimes if bandwidth is small, number of cluster can be larger than 50,
    but we would like to keep max clusters 50 as it is the max number in our dataset.
    In that case you increase the quantile to increase the bandwidth to decrease
    the number of clusters.
    """
    ms = MeanShift()
    while True:
        _, center, bandwidth, cluster_ids = ms.mean_shift(
            embedding, 10000, quantile, iterations, kernel_type=kernel_type
        )
        if torch.unique(cluster_ids).shape[0] > 49:
            quantile *= 1.2
        else:
            break
    return center, bandwidth, cluster_ids


def normalize_points(points):
    EPS = np.finfo(np.float32).eps
    points = points - np.mean(points, 0, keepdims=True)
    S, U = pca_numpy(points)
    smallest_ev = U[:, np.argmin(S)]
    R = rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))
    points = (R @ points.T).T
    std = np.max(points, 0) - np.min(points, 0)
    points = points / (np.max(std) + EPS)
    return points.astype(np.float32)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="ParseNet Segmentation Prediction")
    parser.add_argument("--path_in", type=str, default="./assets/abc_00470.xyz")
    parser.add_argument("--with_normals", type=bool, default=False)
    cfg = parser.parse_args()

    num_channels = 6 if cfg.with_normals else 3
    pth_path = "./logs/pretrained_models/parsenet.pth" if cfg.with_normals else "./logs/pretrained_models/parsenet_no_normals.pth"

    # Loss = EmbeddingLoss(margin=1.0)
    model = PrimitivesEmbeddingDGCNGn(
        embedding=True,
        emb_size=128,
        primitives=True,
        num_primitives=10,
        # loss_function=Loss.triplet_loss,
        mode=0,
        num_channels=num_channels,
    )
    model = torch.nn.DataParallel(model, device_ids=[0])

    model.to(device)
    model.eval()
    model.load_state_dict(
        torch.load(pth_path)
    )

    iterations = 50
    quantile = 0.015

    points = np.loadtxt(cfg.path_in).astype(np.float32)
    points = normalize_points(points)
    points = torch.from_numpy(points)[None, :].to(device)

    with torch.no_grad():
        embedding, _, _ = model(
            points.permute(0, 2, 1), torch.zeros_like(points)[:, 0], False
        )
    embedding = torch.nn.functional.normalize(embedding[0].T, p=2, dim=1)

    _, _, cluster_ids = guard_mean_shift(
        embedding, quantile, iterations, kernel_type="gaussian"
    )

    cluster_ids = cluster_ids.data.cpu().numpy()

    np.savetxt(
        cfg.path_in.replace(".xyz", "_prediction.xyzc"),
        np.hstack([points.detach().cpu().numpy()[0], cluster_ids[:, None]]),
    )
