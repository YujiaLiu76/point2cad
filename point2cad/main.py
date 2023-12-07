import argparse
import multiprocessing
import numpy as np
import os
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from point2cad.fitting_one_surface import process_one_surface
from point2cad.io_utils import save_unclipped_meshes, save_clipped_meshes, save_topology
from point2cad.utils import seed_everything, continuous_labels, normalize_points, make_colormap_optimal


def process_multiprocessing(cfg, uniq_labels, points, labels, device):
    out_meshes = {}
    with ProcessPoolExecutor(max_workers=cfg.max_parallel_surfaces) as executor:
        futures = {
            executor.submit(process_one_surface, idx, points, labels, cfg, device): idx
            for idx in uniq_labels
        }

        for future in tqdm(
            as_completed(futures), total=len(uniq_labels), desc="Fitting surfaces"
        ):
            idx = futures[future]
            out_meshes[idx] = future.result()
    out_meshes = [out_meshes[idx] for idx in uniq_labels if out_meshes[idx] is not None]
    return out_meshes


def process_singleprocessing(cfg, uniq_labels, points, labels, device):
    out_meshes = []
    for idx in tqdm(uniq_labels, total=len(uniq_labels), desc="Fitting surfaces"):
        surface = process_one_surface(idx, points, labels, cfg, device)
        if surface is not None:
            out_meshes.append(surface)
    return out_meshes


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    color_list = make_colormap_optimal()

    parser = argparse.ArgumentParser(description="Point2CAD pipeline")
    parser.add_argument("--path_in", type=str, default="./assets/abc_00949.xyzc")
    parser.add_argument("--path_out", type=str, default="./out")
    parser.add_argument("--validate_checkpoint_path", type=str, default=None)
    parser.add_argument("--silent", default=True)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--max_parallel_surfaces", type=int, default=4)
    parser.add_argument("--num_inr_fit_attempts", type=int, default=1)
    parser.add_argument("--surfaces_multiprocessing", type=int, default=1)
    cfg = parser.parse_args()

    seed_everything(cfg.seed)

    fn_process = process_singleprocessing
    if cfg.surfaces_multiprocessing:
        multiprocessing.set_start_method("spawn", force=True)
        fn_process = process_multiprocessing

    assert os.path.exists(cfg.path_in), "Input points could not be accessed"
    os.makedirs(cfg.path_out, exist_ok=True)

    os.makedirs("{}/unclipped".format(cfg.path_out), exist_ok=True)
    os.makedirs("{}/clipped".format(cfg.path_out), exist_ok=True)
    os.makedirs("{}/topo".format(cfg.path_out), exist_ok=True)

    # ============================ load points ============================
    points_labels = np.loadtxt(cfg.path_in).astype(np.float32)
    assert (
        points_labels.shape[1] == 4
    ), "This pipeline expects annotated point clouds (4 values per point). Refer to README for further instructions"
    points = points_labels[:, :3]
    labels = points_labels[:, 3].astype(np.int32)
    labels = continuous_labels(labels)

    points = normalize_points(points)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    uniq_labels = np.unique(labels)

    # ============================ fit surfaces ============================

    out_meshes = fn_process(cfg, uniq_labels, points, labels, device)

    # ============================ save unclipped meshes ============================
    print("Saving unclipped meshes...")
    pm_meshes = save_unclipped_meshes(
        out_meshes, color_list, "{}/unclipped/mesh.ply".format(cfg.path_out)
    )

    # ============================ save clipped meshes ==============================
    print("Saving clipped meshes...")
    clipped_meshes = save_clipped_meshes(
        pm_meshes, out_meshes, color_list, "{}/clipped/mesh.ply".format(cfg.path_out)
    )

    # ============================ get edges and corners ============================
    print("Saving topology (edges and corners)...")
    save_topology(clipped_meshes, "{}/topo/topo.json".format(cfg.path_out))

    print("Done")
