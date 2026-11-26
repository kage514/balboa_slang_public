# HW4.7
# uv run python main.py 4_6 scenes/hw4/cirno_fumo_net.py

import json
from pathlib import Path

import numpy as np

from scenes.hw4.geometry import make_cloth_constraints, make_cloth_grid


def make_scene():
    # Release the cloth above the upper sphere, slightly to its left.
    nx, ny = 35, 29
    positions, triangles, dx, dy = make_cloth_grid(nx, ny, 1.43, 1.15)
    positions[:, 1] = 5.12 + 0.06 * positions[:, 0]
    positions[:, 0] -= 0.53
    edges, lengths = make_cloth_constraints(nx, ny, dx, dy)

    # The inclined side plane guides the cloth back toward the lower sphere.
    guide = np.array([1.0, 0.11, 0.0, -0.82]) / np.hypot(1.0, 0.11)
    planes = np.array([[0, 1, 0, 0], guide], dtype=np.float32)
    spheres = np.array([
        [0.43, 3.23, 0.0, 0.70],
        [-1.07, 1.84, 0.0, 0.47],
    ], dtype=np.float32)

    filename = Path(__file__).with_name("cirno_fumo_assets") / "cirno_mesh.json"
    mesh = json.loads(filename.read_text(encoding="utf-8"))
    vertices = np.asarray(mesh["vertices"], dtype=np.float32)
    normals = np.asarray(mesh["vertex_normals"], dtype=np.float32)
    # Stand Fumo under the path off the lower sphere.
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    center[1] = vertices[:, 1].min()
    vertices = (vertices - center) * np.float32(1.64 / np.ptp(vertices[:, 1]))
    angle = -0.28
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    vertices = vertices @ rotation.T + np.array([-0.16, 0.0, -0.14], dtype=np.float32)

    return {
        "positions": positions,
        "inv_masses": np.ones(len(positions), dtype=np.float32),
        "constraint_edges": edges,
        "rest_lengths": lengths,
        "triangles": triangles,
        "cloth_grid_size": (nx, ny),
        "spacing": (dx, dy),
        "planes": planes,
        "spheres": spheres,
        "collision_mesh": {
            "vertices": vertices,
            "vertex_normals": normals @ rotation.T,
            "triangles": np.asarray(mesh["faces"], dtype=np.uint32),
        },
        "collision_distance": 0.022,
        "contact_margin": 0.065,
        "static_friction": 0.72,
        "kinetic_friction": 0.53,
        "substeps": 8,
        "iterations": 18,
        "cam_pos": (3.15, 4.16, -5.55),
        "cam_target": (-0.28, 2.40, -0.05),
        "cam_up": (0.0, 1.0, 0.0),
    }

