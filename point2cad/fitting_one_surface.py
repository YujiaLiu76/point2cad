import numpy as np
import open3d as o3d
import os
import pyvista as pv
import torch
import warnings
from tqdm import tqdm

from dependencies.geomfitty.geomfitty._util import distance_line_point
from point2cad.fitting_utils import project_to_plane, visualize_basic_mesh
from point2cad.layers import PositionalEncoding, ResBlock, SirenLayer, SirenWithResblock
from point2cad.primitive_forward import Fit
from point2cad.utils import get_rng
from point2cad.utils import sample_inr_mesh


def process_one_surface(label, points, labels, cfg, device):
    in_points = points[labels == label]
    if len(in_points) < 20:
        return None

    in_points = torch.from_numpy(in_points).to(device)

    # ========================= fitting basic primitives =======================
    recon_basic_shapes = fit_basic_primitives(in_points)

    # ========================== fitting inrs=====================
    recon_inr_shapes = fit_inrs(in_points, cfg, device)

    # ==========================shape selection====================
    pred_info = {}
    pred_info["id"] = int(label)

    if "cone_failure" not in recon_basic_shapes.keys():
        cone_err = np.inf
    elif recon_basic_shapes["cone_failure"]:
        cone_err = np.inf
    else:
        cone_err = recon_basic_shapes["cone_err"]

    plane_err = recon_basic_shapes["plane_err"]
    sphere_err = recon_basic_shapes["sphere_err"]
    cylinder_err = recon_basic_shapes["cylinder_err"]

    if (
        visualize_basic_mesh("cone", in_points, recon_basic_shapes, device=device)
        is None
    ):
        cone_err = np.inf
    elif (
        len(
            visualize_basic_mesh(
                "cone", in_points, recon_basic_shapes, device=device
            ).vertices
        )
        == 0
    ):
        cone_err = np.inf
    if "cone_params" not in recon_basic_shapes.keys():
        cone_err = np.inf
    else:
        if recon_basic_shapes["cone_params"][2] >= 1.53:
            cone_err = np.inf
    if (
        len(
            visualize_basic_mesh(
                "cylinder", in_points, recon_basic_shapes, device=device
            ).vertices
        )
        == 0
    ):
        cylinder_err = np.inf
    if (
        len(
            visualize_basic_mesh(
                "sphere", in_points, recon_basic_shapes, device=device
            ).vertices
        )
        == 0
    ):
        sphere_err = np.inf

    if recon_inr_shapes["is_good_fit"]:
        inr_err = recon_inr_shapes["err"]
    else:
        inr_err = np.inf

    all_errors = np.array([plane_err, sphere_err, cylinder_err, cone_err, inr_err])
    sorted_shape_indices = np.argsort(
        [plane_err, sphere_err, cylinder_err, cone_err, inr_err]
    )
    min_indices_tmp = sorted_shape_indices[0]

    preference_basic_error_increment_thres = 0.001
    preference_basic_error_thres = 0.008
    if min_indices_tmp == 4:
        if (
            np.min([plane_err, sphere_err, cylinder_err, cone_err])
            < preference_basic_error_thres
        ):
            pred_shape = np.argmin([plane_err, sphere_err, cylinder_err, cone_err])
        else:
            if (
                np.min([plane_err, sphere_err, cylinder_err, cone_err])
                < all_errors[min_indices_tmp] + preference_basic_error_increment_thres
            ):
                pred_shape = np.argmin([plane_err, sphere_err, cylinder_err, cone_err])
            else:
                pred_shape = min_indices_tmp
    else:
        pred_shape = min_indices_tmp

    if pred_shape == 0:  # plane
        pred_mesh = visualize_basic_mesh(
            "plane", in_points, recon_basic_shapes, device=device
        )
        pred_mesh.triangle_normals = o3d.utility.Vector3dVector([])
        o3d.io.write_triangle_mesh("tmp.obj", pred_mesh)
        pred_mesh = pv.read("tmp.obj")
        os.remove("tmp.obj")
        pred_info["type"] = "plane"
        pred_info["params"] = recon_basic_shapes["plane_params"]
        pred_info["err"] = plane_err

    elif pred_shape == 1:  # sphere
        pred_mesh = visualize_basic_mesh(
            "sphere", in_points, recon_basic_shapes, device=device
        )
        pred_mesh.triangle_normals = o3d.utility.Vector3dVector([])
        o3d.io.write_triangle_mesh("tmp.obj", pred_mesh)
        pred_mesh = pv.read("tmp.obj")
        os.remove("tmp.obj")
        pred_info["type"] = "sphere"
        pred_info["params"] = recon_basic_shapes["sphere_params"]
        pred_info["err"] = sphere_err

    elif pred_shape == 2:  # cylinder
        pred_mesh = visualize_basic_mesh(
            "cylinder", in_points, recon_basic_shapes, device=device
        )
        pred_mesh.triangle_normals = o3d.utility.Vector3dVector([])
        o3d.io.write_triangle_mesh("tmp.obj", pred_mesh)
        pred_mesh = pv.read("tmp.obj")
        os.remove("tmp.obj")
        pred_info["type"] = "cylinder"
        pred_info["params"] = recon_basic_shapes["cylinder_params"]
        pred_info["err"] = cylinder_err

    elif pred_shape == 3:  # cone
        if np.abs(plane_err - cone_err) > 1e-5:
            pred_mesh = visualize_basic_mesh(
                "cone", in_points, recon_basic_shapes, device=device
            )
            pred_mesh.triangle_normals = o3d.utility.Vector3dVector([])
            o3d.io.write_triangle_mesh("tmp.obj", pred_mesh)
            pred_mesh = pv.read("tmp.obj")
            os.remove("tmp.obj")
            pred_info["type"] = "cone"
            pred_info["params"] = recon_basic_shapes["cone_params"]
            pred_info["err"] = cone_err

        else:
            pred_mesh = visualize_basic_mesh(
                "plane", in_points, recon_basic_shapes, device=device
            )
            pred_mesh.triangle_normals = o3d.utility.Vector3dVector([])
            o3d.io.write_triangle_mesh("tmp.obj", pred_mesh)
            pred_mesh = pv.read("tmp.obj")
            os.remove("tmp.obj")
            pred_info["type"] = "plane"
            pred_info["params"] = recon_basic_shapes["plane_params"]
            pred_info["err"] = plane_err

    elif pred_shape == 4:
        pred_mesh = pv.wrap(recon_inr_shapes["mesh_uv"])
        pred_info["type"] = "open_spline"
        pred_info["err"] = inr_err
        pred_info["params"] = None

    if pred_mesh.n_points > 0:
        out = {}
        out["mesh"] = pred_mesh
        out["info"] = pred_info
        out["inpoints"] = in_points.detach().cpu().numpy()

    return out


def fit_basic_primitives(pts):
    """
    output: a dict of reconstructed points of each fitting shape, residual error

    """
    if pts.shape[0] < 20:
        raise ValueError("the number of points in the patch is too small")

    fitting = Fit()
    recon_basic_shapes = {}

    # ==================fit a plane=========================
    axis, distance = fitting.fit_plane_torch(
        points=pts,
        normals=None,
        weights=torch.ones_like(pts)[:, :1],
        ids=None,
    )
    # Project points on the surface
    new_points = project_to_plane(pts, axis, distance.item())
    plane_err = torch.linalg.norm(new_points - pts, dim=-1).mean()

    new_points = fitting.sample_plane(
        distance.item(),
        axis.data.cpu().numpy(),
        mean=torch.mean(new_points, 0).data.cpu().numpy(),
    )
    recon_basic_shapes["plane_params"] = (
        axis.data.cpu().numpy().tolist(),
        distance.data.cpu().numpy().tolist(),
    )
    recon_basic_shapes["plane_new_points"] = new_points.tolist()
    recon_basic_shapes["plane_err"] = plane_err.data.cpu().numpy().tolist()

    # ======================fit a sphere======================
    center, radius = fitting.fit_sphere_torch(
        pts,
        normals=None,
        weights=torch.ones_like(pts)[:, :1],
        ids=None,
    )
    sphere_err = (torch.linalg.norm(pts - center, dim=-1) - radius).abs().mean()

    # Project points on the surface
    new_points, new_normals = fitting.sample_sphere(
        radius.item(), center.data.cpu().numpy(), N=10000
    )
    center = center.data.cpu().numpy()

    recon_basic_shapes["sphere_params"] = (center.tolist(), radius.tolist())
    recon_basic_shapes["sphere_new_points"] = new_points.tolist()
    recon_basic_shapes["sphere_err"] = sphere_err.data.cpu().numpy().tolist()

    # ======================fit a cylinder====================
    a, center, radius = fitting.fit_cylinder(
        pts,
        normals=torch.zeros_like(pts),
        weights=torch.ones_like(pts)[:, :1],
        ids=None,
    )

    new_points, new_normals = fitting.sample_cylinder_trim(
        radius.item(),
        center,
        a,
        pts.data.cpu().numpy(),
        N=10000,
    )
    cylinder_err = np.abs(
        (distance_line_point(center, a, pts.detach().cpu().numpy()) - radius)
    ).mean()

    recon_basic_shapes["cylinder_params"] = (
        a.tolist(),
        center.tolist(),
        radius.tolist(),
    )
    recon_basic_shapes["cylinder_new_points"] = new_points.tolist()
    recon_basic_shapes["cylinder_err"] = cylinder_err.tolist()

    # ==========================fit a cone======================
    apex, axis, theta, cone_err, failure = fitting.fit_cone(
        pts,
        normals=torch.zeros_like(pts),
        weights=torch.ones_like(pts)[:, :1],
        ids=None,
    )
    new_points, new_normals = fitting.sample_cone_trim(
        apex, axis, theta, pts.data.cpu().numpy()
    )
    if new_normals is not None:
        recon_basic_shapes["cone_params"] = (
            apex.tolist(),
            axis.tolist(),
            theta.tolist(),
        )
        recon_basic_shapes["cone_new_points"] = new_points.tolist()
        recon_basic_shapes["cone_failure"] = failure
        recon_basic_shapes["cone_err"] = cone_err.tolist()

    return recon_basic_shapes


def fit_inrs(pts, cfg, device="cuda"):
    def fit_one_inr_wrapper(is_u_closed, is_v_closed, seed):
        return fit_one_inr_spline_config(
            pts,
            is_u_closed=is_u_closed,
            is_v_closed=is_v_closed,
            model_dim_hidden=64,
            model_num_hidden_layers=0,
            model_block_type="combined",
            model_resblock_posenc_numfreqs=0,
            model_resblock_zeroinit_posenc=True,
            model_resblock_act_type="silu",
            model_resblock_batchnorms=False,
            model_resblock_shortcut=False,
            model_resblock_channels_fraction=0.5,
            model_sirenblock_omega_first=10,
            model_sirenblock_omega_other=10,
            model_sirenblock_act_type="sinc",
            model_init_checkpoint_path=cfg.validate_checkpoint_path,
            optimizer="adam",
            optimizer_kwargs=dict(),
            langevin_noise_magnitude_3d=0.005,
            langevin_noise_magnitude_uv=0.005,
            lr=1e-1,
            lr_warmup_steps="auto",
            lr_decay_steps="auto",
            lr_decay_rate=0.001,
            loss_fit_type="l1",
            loss_uv_tightness_weight=0.0,
            loss_uv_tightness_margin=0.2,
            loss_metric_weight=0.0,
            loss_metric_num_samples="auto",
            loss_metric_margin=0.2,
            dtype="fp32",
            num_fit_steps=1000 if cfg.validate_checkpoint_path is None else 0,
            batch_sz="auto",
            batch_sz_schedule="const",
            data_whitening_isometric=True,
            val_split_pct=10,
            good_fit_l2_tol=1e-4,
            device=device,
            seed=seed,
            progress_bar=not cfg.silent,
        )

    out_inr = None
    for s in range(cfg.num_inr_fit_attempts):
        cur_inr = fit_one_inr_wrapper(is_u_closed=False, is_v_closed=False, seed=cfg.seed + s)
        if out_inr is None or cur_inr['err'] < out_inr['err']:
            out_inr = cur_inr

        cur_inr = fit_one_inr_wrapper(is_u_closed=False, is_v_closed=True, seed=cfg.seed + s)
        if out_inr is None or cur_inr['err'] < out_inr['err']:
            out_inr = cur_inr

        cur_inr = fit_one_inr_wrapper(is_u_closed=True, is_v_closed=False, seed=cfg.seed + s)
        if out_inr is None or cur_inr['err'] < out_inr['err']:
            out_inr = cur_inr

        cur_inr = fit_one_inr_wrapper(is_u_closed=True, is_v_closed=True, seed=cfg.seed + s)
        if out_inr is None or cur_inr['err'] < out_inr['err']:
            out_inr = cur_inr

    out_inr["mesh_uv"] = sample_inr_mesh(out_inr, mesh_dim=100, uv_margin=0.2)
    return out_inr


def fit_one_inr_spline_config(points, **kwargs):
    out = fit_one_inr_spline(points, **kwargs)
    kwargs.pop("is_u_closed")
    kwargs.pop("is_v_closed")
    kwargs.pop("model_init_checkpoint_path")
    kwargs.pop("progress_bar")
    out["config"] = dict(**kwargs)
    return out


def fit_one_inr_spline(
    points,
    is_u_closed=False,
    is_v_closed=False,
    model_dim_hidden=64,
    model_num_hidden_layers=1,
    model_block_type="residual",
    model_resblock_posenc_numfreqs=8,
    model_resblock_zeroinit_posenc=True,
    model_resblock_act_type="silu",
    model_resblock_batchnorms=False,
    model_resblock_shortcut=False,
    model_resblock_channels_fraction=0.5,
    model_sirenblock_omega_first=10,
    model_sirenblock_omega_other=10,
    model_sirenblock_act_type="sinc",
    model_init_checkpoint_path=None,
    optimizer="adam",
    optimizer_kwargs=None,
    langevin_noise_magnitude_3d=0.0,
    langevin_noise_magnitude_uv=0.0,
    lr=1e-2,
    lr_warmup_steps="auto",
    lr_decay_steps="auto",
    lr_decay_rate=0.001,
    loss_fit_type="l1",
    loss_uv_tightness_weight=0.0,
    loss_uv_tightness_margin=0.2,
    loss_metric_weight=1.0,
    loss_metric_num_samples="auto",
    loss_metric_margin=0.2,
    dtype="fp32",
    num_fit_steps=1000,
    batch_sz="auto",
    batch_sz_schedule="const",
    data_whitening_isometric=True,
    val_split_pct=10,
    good_fit_l2_tol=1e-4,
    device="cuda",
    seed=None,
    progress_bar=True,
):
    if not torch.is_tensor(points):
        raise ValueError("Input must be a torch tensor")
    if points.dtype not in (torch.float16, torch.float32, torch.float64):
        raise ValueError("Input must be a tensor of floats")
    if points.dim() != 2 or points.shape[0] < 3:
        raise ValueError("Input must be a 2D-array of at least three points")
    if points.shape[1] != 3:
        raise ValueError("Points must be 3D")
    if batch_sz_schedule not in ("const", "linear"):
        raise ValueError(f"Unknown batch_sz_schedule={batch_sz_schedule}")

    if lr_warmup_steps == "auto":
        lr_warmup_steps = num_fit_steps // 20
    if lr_decay_steps == "auto":
        lr_decay_steps = num_fit_steps

    dtype = {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "fp64": torch.float64,
    }[dtype]

    if device != "cpu" and not torch.cuda.is_available():
        warnings.warn("CUDA not available, fitting on CPU may be slow")
        device = "cpu"

    model = SplineINR(
        is_u_closed=is_u_closed,
        is_v_closed=is_v_closed,
        dim_hidden=model_dim_hidden,
        num_hidden_layers=model_num_hidden_layers,
        block_type=model_block_type,
        resblock_posenc_numfreqs=model_resblock_posenc_numfreqs,
        resblock_zeroinit_posenc=model_resblock_zeroinit_posenc,
        resblock_act_type=model_resblock_act_type,
        resblock_batchnorms=model_resblock_batchnorms,
        resblock_shortcut=model_resblock_shortcut,
        resblock_channels_fraction=model_resblock_channels_fraction,
        sirenblock_omega_first=model_sirenblock_omega_first,
        sirenblock_omega_other=model_sirenblock_omega_other,
        sirenblock_act_type=model_sirenblock_act_type,
        dtype=dtype,
    )

    if model_init_checkpoint_path is not None:
        model.load_state_dict(torch.load(model_init_checkpoint_path))

    optimizer = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
        "nadam": torch.optim.NAdam,
        "radam": torch.optim.RAdam,
    }[optimizer](
        model.parameters(),
        lr=lr,
        **(optimizer_kwargs if optimizer_kwargs is not None else {}),
    )

    loss_fit_fn = {
        "l1": torch.nn.L1Loss(),
        "l2": torch.nn.MSELoss(),
        "huber": torch.nn.HuberLoss(),
    }[loss_fit_type]

    num_points = points.shape[0]
    num_points_val = val_split_pct * num_points // 100
    num_points_train = num_points - num_points_val

    if batch_sz == "auto":
        batch_sz = num_points_train
    if loss_metric_num_samples == "auto":
        loss_metric_num_samples = batch_sz // 2

    points = points.to(device)
    points_mean = points.mean(dim=0)
    points_scale = points.std(dim=0)
    if data_whitening_isometric:
        points_scale = points_scale.max()
    points = (points - points_mean) / points_scale
    points_mean = points_mean.cpu()
    points_scale = points_scale.cpu()
    if data_whitening_isometric:
        points_scale = points_scale.item()

    rng_train_common = get_rng(device, seed=seed)
    rng_train_synthetic_uv = get_rng(device, seed=seed, seed_increment=1)

    permutation = torch.randperm(num_points, device=device, generator=rng_train_common)
    points = points[permutation]
    points_train = points[:-num_points_val]
    points_val = points[-num_points_val:]

    points_train_cdist = None
    if num_points_train <= 2048:
        points_train_cdist = torch.cdist(points_train, points_train)

    model = model.to(device)
    model.train()

    pbar = tqdm(range(num_fit_steps), disable=not progress_bar)
    for step in pbar:
        cur_batch_sz = batch_sz
        if batch_sz_schedule == "linear":
            cur_batch_sz = (
                num_points_train * step + batch_sz * (num_fit_steps - step)
            ) // num_fit_steps

        permutation = torch.randperm(
            num_points_train, device=device, generator=rng_train_common
        )
        inds = permutation[:cur_batch_sz]
        x = points_train[inds]

        new_lrate = lr * (lr_decay_rate ** (step / lr_decay_steps))
        if lr_warmup_steps > 0 and step < lr_warmup_steps:
            new_lrate *= step / lr_warmup_steps
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lrate

        optimizer.zero_grad()

        langevin_noise_schedule = (num_fit_steps - step - 1) / (num_fit_steps - 1)
        x_input = x
        if langevin_noise_magnitude_3d > 0:
            x_input = x + (
                langevin_noise_magnitude_3d * langevin_noise_schedule
            ) * torch.randn_like(x)
        uv = model.encoder(x_input)
        uv_input = uv
        if langevin_noise_magnitude_uv > 0:
            uv_input = uv + (
                langevin_noise_magnitude_uv * langevin_noise_schedule
            ) * torch.randn_like(uv)
        x_hat = model.decoder(uv_input)

        synthetic_uv = None
        if loss_uv_tightness_weight > 0:
            uv_scale = 2.0
            uv_tol = 0.01
            uv_scale_vec = torch.tensor(
                [
                    uv_scale if is_u_closed else uv_scale - uv_tol,
                    uv_scale if is_v_closed else uv_scale - uv_tol,
                ],
                dtype=dtype,
                device=device,
            )
            synthetic_uv = (
                torch.rand(
                    *uv.shape, device=uv.device, generator=rng_train_synthetic_uv
                )
                - 0.5
            ) * uv_scale_vec

        loss = loss_fit_fn(x_hat, x)

        pbar_desc = {"fit": loss.item()}

        if loss_uv_tightness_weight > 0:
            with torch.no_grad():
                synthetic_3d = model.decoder(synthetic_uv)
            cdist_synth_x = torch.cdist(synthetic_3d, x)
            nearest_dists, nearest_ids = torch.topk(
                cdist_synth_x, 1, largest=False, sorted=False
            )
            nearest_3d = x[nearest_ids.squeeze(1)]
            nearest_uv = model.encoder(nearest_3d)
            mask_loss_active = nearest_dists > loss_uv_tightness_margin
            loss_uv_tightness = torch.nn.functional.l1_loss(
                nearest_uv, synthetic_uv, reduction="none"
            )
            loss_uv_tightness = (loss_uv_tightness * mask_loss_active).mean()
            loss += loss_uv_tightness_weight * loss_uv_tightness
            pbar_desc["tightness"] = loss_uv_tightness.item()

        if loss_metric_weight > 0:
            if points_train_cdist is None:
                cdist_3d = torch.cdist(x, x)
            else:
                cdist_3d = points_train_cdist[inds, :][:, inds]
            uv_lifted = convert_uv_to_decoder_input(
                uv, is_u_closed, is_v_closed, open_replicate=False
            )
            cdist_uv = torch.cdist(uv_lifted, uv_lifted)
            _, cdist_3d_inds_neg = torch.topk(
                cdist_3d, loss_metric_num_samples, largest=True, sorted=False
            )
            _, cdist_3d_botk_pos = torch.topk(
                cdist_3d, loss_metric_num_samples, largest=False, sorted=False
            )
            cdist_2d_vals_neg = cdist_uv.gather(1, cdist_3d_inds_neg)
            cdist_2d_vals_pos = cdist_uv.gather(1, cdist_3d_botk_pos)
            loss_metric = torch.nn.functional.relu(
                cdist_2d_vals_pos - cdist_2d_vals_neg + loss_metric_margin
            ).mean()
            loss += loss_metric_weight * loss_metric
            pbar_desc["metric"] = loss_metric.item()

        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            pbar.set_description(
                ",".join(f"{k}={v:5.4f}" for k, v in pbar_desc.items())
            )

    model.eval()
    uv_bb_min, uv_bb_max = extract_one_inr_spline_bbox(model, points)
    val_out = val_one_inr_spline(model, points_val)
    is_good_fit = val_out < good_fit_l2_tol
    err = fit_err(model, points)
    model = model.cpu()

    out = {
        "is_u_closed": is_u_closed,
        "is_v_closed": is_v_closed,
        "points3d_offset": points_mean,
        "points3d_scale": points_scale,
        "val_err_l2": val_out,
        "is_good_fit": is_good_fit,
        "uv_bb_min": uv_bb_min,
        "uv_bb_max": uv_bb_max,
        "model": model,
        "err": err,
    }

    return out


class Mapping(torch.nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        dim_hidden=32,
        num_hidden_layers=0,
        block_type="residual",
        resblock_posenc_numfreqs=0,
        resblock_zeroinit_posenc=True,
        resblock_act_type="silu",
        resblock_batchnorms=True,
        resblock_shortcut=False,
        resblock_channels_fraction=0.5,
        sirenblock_omega_first=10,
        sirenblock_omega_other=10,
        sirenblock_act_type="sinc",
        dtype=torch.float32,
        checks=True,
    ):
        super().__init__()
        self.dtype = dtype
        if block_type == "residual":
            posenc = PositionalEncoding(resblock_posenc_numfreqs, True, dtype)
            dim_in_real = dim_in * posenc.dim_multiplier
            layers = [
                posenc,
                ResBlock(
                    dim_in_real,
                    dim_hidden,
                    batchnorms=resblock_batchnorms,
                    act_type=resblock_act_type,
                    shortcut=False,
                ),
            ]
            layers += [
                ResBlock(
                    dim_hidden,
                    dim_hidden,
                    batchnorms=resblock_batchnorms,
                    act_type=resblock_act_type,
                    shortcut=resblock_shortcut,
                )
            ] * num_hidden_layers
            layers += [torch.nn.Linear(dim_hidden, dim_out)]
            if resblock_zeroinit_posenc:
                layer_init = torch.nn.Linear(dim_in, dim_hidden)
                with torch.no_grad():
                    layers[1].linear.weight *= 0.01
                    layers[1].linear.weight[:, :dim_in].copy_(layer_init.weight)
        elif block_type == "siren":
            layers = [
                SirenLayer(
                    dim_in,
                    dim_hidden,
                    is_first=True,
                    omega=sirenblock_omega_first,
                    act_type=sirenblock_act_type,
                )
            ]
            layers += [
                SirenLayer(
                    dim_hidden,
                    dim_hidden,
                    is_first=False,
                    omega=sirenblock_omega_other,
                    act_type=sirenblock_act_type,
                )
            ] * num_hidden_layers
            layers += [torch.nn.Linear(dim_hidden, dim_out)]
        elif block_type == "combined":
            layers = [
                SirenWithResblock(
                    dim_in,
                    dim_hidden,
                    sirenblock_is_first=True,
                    sirenblock_omega=sirenblock_omega_first,
                    sirenblock_act_type=sirenblock_act_type,
                    resblock_batchnorms=resblock_batchnorms,
                    resblock_act_type=resblock_act_type,
                    resblock_shortcut=resblock_shortcut,
                    resblock_channels_fraction=resblock_channels_fraction,
                )
            ]
            layers += [
                SirenWithResblock(
                    dim_hidden,
                    dim_hidden,
                    sirenblock_is_first=False,
                    sirenblock_omega=sirenblock_omega_other,
                    sirenblock_act_type=sirenblock_act_type,
                    resblock_batchnorms=resblock_batchnorms,
                    resblock_act_type=resblock_act_type,
                    resblock_shortcut=resblock_shortcut,
                    resblock_channels_fraction=resblock_channels_fraction,
                )
            ] * num_hidden_layers
            layers += [torch.nn.Linear(dim_hidden, dim_out)]
        else:
            raise ValueError(f"Unknown block_type={block_type}")
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SplineINR(torch.nn.Module):
    def __init__(self, is_u_closed=False, is_v_closed=False, **kwargs):
        super().__init__()
        self.encoder = Map3DtoUV(is_u_closed, is_v_closed, **kwargs)
        self.decoder = MapUVto3D(is_u_closed, is_v_closed, **kwargs)

    def forward(self, x):
        uv = self.encoder(x)
        x_hat = self.decoder(uv)
        return x_hat, uv


def convert_encoder_output_to_uv(x, is_u_closed, is_v_closed):
    xu, xv = x.chunk(2, dim=1)  # B x 2, B x 2
    if is_u_closed:
        xu = torch.atan2(xu[:, [0]], xu[:, [1]]) / np.pi  # B x 1
    else:
        xu = torch.tanh(xu[:, [0]])  # B x 1
    if is_v_closed:
        xv = torch.atan2(xv[:, [0]], xv[:, [1]]) / np.pi  # B x 1
    else:
        xv = torch.tanh(xv[:, [0]])  # B x 1
    x = torch.cat((xu, xv), dim=1)  # B x 2
    return x  # B x 2


def convert_uv_to_decoder_input(x, is_u_closed, is_v_closed, open_replicate=True):
    if is_u_closed:
        xu_closed_rad = x[:, [0]] * np.pi  # B x 1
        xu_0 = xu_closed_rad.cos()
        xu_1 = xu_closed_rad.sin()
    else:
        xu_open = x[:, [0]]  # B x 1
        xu_0 = xu_open
        xu_1 = xu_open if open_replicate else torch.zeros_like(xu_open)
    if is_v_closed:
        xv_closed_rad = x[:, [1]] * np.pi  # B x 2
        xv_0 = xv_closed_rad.cos()
        xv_1 = xv_closed_rad.sin()
    else:
        xv_open = x[:, [1]]  # B x 2
        xv_0 = xv_open
        xv_1 = xv_open if open_replicate else torch.zeros_like(xv_open)
    x = torch.cat((xu_0, xu_1, xv_0, xv_1), dim=1)  # B x 4
    return x  # B x 4


class Map3DtoUV(Mapping):
    def __init__(self, is_u_closed, is_v_closed, **kwargs):
        self.is_u_closed = is_u_closed
        self.is_v_closed = is_v_closed
        super().__init__(3, 4, **kwargs)

    def forward(self, x):
        x = super().forward(x)
        x = convert_encoder_output_to_uv(x, self.is_u_closed, self.is_v_closed)
        return x


class MapUVto3D(Mapping):
    def __init__(self, is_u_closed, is_v_closed, **kwargs):
        self.is_u_closed = is_u_closed
        self.is_v_closed = is_v_closed
        super().__init__(4, 3, **kwargs)

    def forward(self, x):
        if not torch.is_tensor(x) or x.dim() not in (1, 2) or x.dtype != self.dtype:
            raise ValueError(f"Invalid input")
        if x.dim() == 1:
            is_batch_dim_unsqueezed = True
            x = x.unsqueeze(0)
        else:
            is_batch_dim_unsqueezed = False
        x = convert_uv_to_decoder_input(x, self.is_u_closed, self.is_v_closed)
        x = super().forward(x)
        if is_batch_dim_unsqueezed:
            x = x.squeeze(0)
        return x


def extract_one_inr_spline_bbox(model, points=None, uv=None):
    if uv is None:
        with torch.no_grad():
            uv = model.encoder(points)
    uv_bb_min = uv.min(dim=0).values.cpu().detach()
    uv_bb_max = uv.max(dim=0).values.cpu().detach()
    return uv_bb_min, uv_bb_max


def val_one_inr_spline(model, points):
    with torch.no_grad():
        x_hat, _ = model(points)
        val_l2 = torch.nn.functional.mse_loss(x_hat, points).item()
    return val_l2


def fit_err(model, points):
    with torch.no_grad():
        x_hat, _ = model(points)
        err = torch.sqrt(((points - x_hat) ** 2).sum(-1)).mean()
    return err.item()
