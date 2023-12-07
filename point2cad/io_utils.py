import itertools
import json
import numpy as np
import pymesh
import pyvista as pv
import scipy
import trimesh
from collections import Counter

from point2cad.utils import suppress_output_fd


def save_unclipped_meshes(meshes, color_list, out_path):
    non_clipped_meshes = []
    pm_meshes = []
    for s in range(len(meshes)):
        tri_meshes_s = trimesh.Trimesh(
            vertices=np.array(meshes[s]["mesh"].points),
            faces=np.array(meshes[s]["mesh"].faces.reshape(-1, 4)[:, 1:]),
        )
        tri_meshes_s.visual.face_colors = color_list[s]
        non_clipped_meshes.append(tri_meshes_s)
        pm_meshes.append(
            pymesh.form_mesh(
                meshes[s]["mesh"].points,
                meshes[s]["mesh"].faces.reshape(-1, 4)[:, 1:],
            )
        )

    final_non_clipped = trimesh.util.concatenate(non_clipped_meshes)
    final_non_clipped.export(out_path)

    return pm_meshes


def save_clipped_meshes(pm_meshes, out_meshes, color_list, out_path):
    pm_merged = pymesh.merge_meshes(pm_meshes)

    face_sources_merged = pm_merged.get_attribute("face_sources").astype(np.int32)

    detect_pairs = pymesh.detect_self_intersection(pm_merged)
    pm_resolved_ori = pymesh.resolve_self_intersection(pm_merged)

    a = pymesh.separate_mesh(pm_resolved_ori)

    pm_resolved, info_dict = pymesh.remove_duplicated_vertices(
        pm_resolved_ori, tol=1e-6, importance=None
    )

    face_sources_resolved_ori = pm_resolved_ori.get_attribute("face_sources").astype(
        np.int32
    )
    face_sources_from_fit = face_sources_merged[face_sources_resolved_ori]

    tri_resolved = trimesh.Trimesh(
        vertices=pm_resolved.vertices, faces=pm_resolved.faces
    )
    face_adjacency = tri_resolved.face_adjacency

    connected_node_labels = trimesh.graph.connected_component_labels(
        edges=face_adjacency, node_count=len(tri_resolved.faces)
    )

    most_common_groupids = [
        item[0] for item in Counter(connected_node_labels).most_common()
    ]

    submeshes = [
        trimesh.Trimesh(
            vertices=np.array(tri_resolved.vertices),
            faces=np.array(tri_resolved.faces)[np.where(connected_node_labels == item)],
        )
        for item in most_common_groupids
    ]
    indices_sources = [
        face_sources_from_fit[connected_node_labels == item][0]
        for item in np.array(most_common_groupids)
    ]

    clipped_meshes = []
    further_clipped_meshes = []
    for p in range(len(out_meshes)):
        one_cluter_points = out_meshes[p]["inpoints"]
        submeshes_cur = [
            x
            for x, y in zip(submeshes, np.array(indices_sources) == p)
            if y and len(x.faces) > 2
        ]
        nearest_submesh = np.argmin(
            np.array(
                [
                    trimesh.proximity.closest_point(item, one_cluter_points)[1]
                    for item in submeshes_cur
                ]
            ).transpose(),
            -1,
        )
        counter_nearest = Counter(nearest_submesh).most_common()
        area_per_point = np.array(
            [submeshes_cur[item[0]].area / item[1] for item in counter_nearest]
        )

        multiplier_area = 2
        result_indices = np.array(counter_nearest)[:, 0][
            np.logical_and(
                area_per_point
                < area_per_point[np.nonzero(area_per_point)[0][0]] * multiplier_area,
                area_per_point != 0,
            )
        ]

        result_submesh_list = [submeshes_cur[item] for item in result_indices]

        clipped_mesh = trimesh.util.concatenate(result_submesh_list)
        clipped_mesh.visual.face_colors = color_list[p]
        clipped_meshes.append(clipped_mesh)

    clipped = trimesh.util.concatenate(clipped_meshes)
    clipped.export(out_path)

    return clipped_meshes


def save_topology(clipped_meshes, out_path):
    filtered_submeshes_pv = [pv.wrap(item) for item in clipped_meshes]

    filtered_submeshes_pv_combinations = list(
        itertools.combinations(filtered_submeshes_pv, 2)
    )
    intersected_pair_indices = []
    intersection_curves = []
    intersections = {}

    for k, pv_pair in enumerate(filtered_submeshes_pv_combinations):
        with suppress_output_fd():
            intersection, _, _ = pv_pair[0].intersection(
                pv_pair[1], split_first=False, split_second=False, progress_bar=False
            )
        if intersection.n_points > 0:
            intersected_pair_indices.append(k)
            intersection_curve = {}
            intersection_curve["pv_points"] = intersection.points.tolist()
            intersection_curve["pv_lines"] = intersection.lines.reshape(-1, 3)[
                :, 1:
            ].tolist()
            intersection_curves.append(intersection_curve)

    intersections["curves"] = intersection_curves

    intersection_corners = []
    intersection_curves_combinations_indices = list(
        itertools.combinations(range(len(intersection_curves)), 2)
    )
    for combination_indices in intersection_curves_combinations_indices:
        sample0 = np.array(intersection_curves[combination_indices[0]]["pv_points"])
        sample1 = np.array(intersection_curves[combination_indices[1]]["pv_points"])

        dists = scipy.spatial.distance.cdist(sample0, sample1)
        row_indices, col_indices = np.where(dists == 0)

        if len(row_indices) > 0 and len(col_indices) > 0:
            corners = [
                (sample0[item[0]] + sample1[item[1]]) / 2
                for item in zip(row_indices, col_indices)
            ]
            intersection_corners.extend(corners)

    intersections["corners"] = [arr.tolist() for arr in intersection_corners]

    with open(out_path, "w") as cf:
        json.dump(intersections, cf)
