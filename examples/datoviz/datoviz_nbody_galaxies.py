"""Native Vulkan Hardware-Rasterized 16,384-Body Colliding Galaxies (vkspatial.BarnesHut3D + Datoviz).

Combines:
- `vkspatial.BarnesHut3D`: 6-level 3D Morton Octree + elastic collisions on Vulkan compute (~1.2 ms/step)
- `datoviz`: Native Vulkan hardware-rasterized 3D point sprites + orbital trail segments + Arcball camera + Dear ImGui HUD

Run interactively (60+ FPS native Vulkan window):
    PYTHONPATH=. pixi run python examples/datoviz/datoviz_nbody_galaxies.py

Run offscreen screenshot test:
    PYTHONPATH=. pixi run python examples/datoviz/datoviz_nbody_galaxies.py --offscreen /tmp/dvz_galaxies.png --steps 140
"""

from __future__ import annotations

import argparse
import ctypes
import math
import sys
from pathlib import Path

_root = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "vknumpy").is_dir()), Path.cwd())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import datoviz as dvz
import numpy as np
import vkspatial as vksp


NUM_BODIES = 16384
WORLD_SCALE = 1.0 / 34.0


def create_colliding_galaxies(n: int = NUM_BODIES, seed: int = 42):
    """Initialize two tilted rotating spiral galaxies (original G=0.025, circular orbits)."""
    rng = np.random.default_rng(seed)
    half = n // 2
    pos_mass = np.zeros((n, 4), dtype=np.float32)
    vel_gal = np.zeros((n, 4), dtype=np.float32)

    G = 0.025
    core_mass = 1800.0
    configs = [
        (slice(0, half), np.array([-17.0, 3.5, -6.0]), np.array([1.25, -0.22, 0.55]), 0.38, 0.0),
        (slice(half, n), np.array([17.0, -3.5, 6.0]), np.array([-1.25, 0.22, -0.55]), -0.48, 1.0),
    ]

    for slc, center, bulk_vel, tilt, gal_id in configs:
        count = slc.stop - slc.start
        u = rng.uniform(0.0, 1.0, count).astype(np.float32)
        radii = (1.2 + 18.0 * (u ** 0.75)).astype(np.float32)
        arm = rng.integers(0, 2, count).astype(np.float32) * np.pi
        angles = (arm + radii * 0.35 + rng.normal(0.0, 0.45, count)).astype(np.float32)
        heights = rng.normal(0.0, 0.35, count).astype(np.float32)

        px, py, pz = radii * np.cos(angles), heights, radii * np.sin(angles)
        enclosed_mass = core_mass + (radii / 19.2) * (count * 0.45)
        orb_speed = np.sqrt(G * enclosed_mass / np.maximum(radii, 0.6))
        vx, vy, vz = -orb_speed * np.sin(angles), np.zeros(count, dtype=np.float32), orb_speed * np.cos(angles)

        c, s = math.cos(tilt), math.sin(tilt)
        py, pz = c * py - s * pz, s * py + c * pz
        vy, vz = c * vy - s * vz, s * vy + c * vz

        pos = np.stack([px, py, pz], axis=-1) + center
        vel = np.stack([vx, vy, vz], axis=-1) + bulk_vel
        mass = rng.uniform(0.2, 0.9, count).astype(np.float32)
        pos[0], vel[0], mass[0] = center, bulk_vel, core_mass

        pos_mass[slc, 0:3] = pos
        pos_mass[slc, 3] = mass
        vel_gal[slc, 0:3] = vel
        vel_gal[slc, 3] = gal_id

    return pos_mass, vel_gal


def compute_star_colors(vel_gal_np: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Compute RGBA uint8 colors: Cyan-Blue vs Crimson-Gold + white-hot collision flashes."""
    n = vel_gal_np.shape[0]
    gal = np.floor(vel_gal_np[:, 3])
    flash = np.clip(vel_gal_np[:, 3] - gal, 0.0, 1.0)
    speed = np.linalg.norm(vel_gal_np[:, :3], axis=1)
    t = np.clip(speed / 5.5, 0.0, 1.0)

    r = np.where(gal < 0.5, 0.18 + 0.72 * t, 1.00)
    g = np.where(gal < 0.5, 0.58 + 0.38 * t, 0.30 + 0.65 * t)
    b = np.where(gal < 0.5, 1.00, 0.18 + 0.25 * (1.0 - t))

    # Boost white-hot collision flashes & supermassive cores
    boost = flash * 0.95
    r = np.clip(r + boost, 0.0, 1.0)
    g = np.clip(g + boost, 0.0, 1.0)
    b = np.clip(b + boost, 0.0, 1.0)

    is_core = masses > 100.0
    r[is_core], g[is_core], b[is_core] = 1.0, 0.96, 0.82

    rgba = np.empty((n, 4), dtype=np.uint8)
    rgba[:, 0] = (r * 255.0).astype(np.uint8)
    rgba[:, 1] = (g * 255.0).astype(np.uint8)
    rgba[:, 2] = (b * 255.0).astype(np.uint8)
    rgba[:, 3] = 175
    rgba[is_core, 3] = 255
    return rgba


def render_datoviz_snapshot(
    pos_mass_np: np.ndarray,
    vel_gal_np: np.ndarray,
    prev_pos_np: np.ndarray | None = None,
    *,
    width: int = 760,
    height: int = 500,
    yaw: float = 0.35,
    pitch: float = 0.55,
    dist: float = 1.75,
    out_path: str = "/tmp/dvz_nbody_snap.png",
) -> np.ndarray:
    """Render a single hardware-rasterized Datoviz Vulkan 3D frame and return RGBA uint8."""
    from PIL import Image

    n = pos_mass_np.shape[0]
    pos3 = np.ascontiguousarray(pos_mass_np[:, :3] * WORLD_SCALE, dtype=np.float32)
    col = compute_star_colors(vel_gal_np, pos_mass_np[:, 3])
    sizes = np.where(pos_mass_np[:, 3] > 100.0, 14.0, 2.0).astype(np.float32)

    app = dvz.app(dvz.APP_FLAGS_OFFSCREEN)
    batch = dvz.app_batch(app)
    scene = dvz.scene(batch)
    fig = dvz.figure(scene, width, height, 0)
    panel = dvz.panel_default(fig)
    dvz.panel_arcball(panel, 0)
    cam = dvz.panel_camera(panel, 0)

    cx = float(dist * math.cos(pitch) * math.sin(yaw))
    cy = float(dist * math.sin(pitch))
    cz = float(dist * math.cos(pitch) * math.cos(yaw))
    dvz.camera_position(cam, dvz.vec3(cx, cy, cz))
    dvz.camera_lookat(cam, dvz.vec3(0.0, 0.0, 0.0))

    # Optional velocity trail segments
    if prev_pos_np is not None:
        p0 = np.ascontiguousarray(prev_pos_np * WORLD_SCALE, dtype=np.float32)
        seg = dvz.segment(batch, 0)
        dvz.segment_alloc(seg, n)
        dvz.segment_position(seg, 0, n, p0, pos3, 0)
        seg_col = col.copy()
        seg_col[:, 3] = 110
        dvz.segment_color(seg, 0, n, seg_col, 0)
        lw = np.full(n, 1.5, dtype=np.float32)
        dvz.segment_linewidth(seg, 0, n, lw, 0)
        dvz.panel_visual(panel, seg, 0)

    pts = dvz.point(batch, 0)
    dvz.point_alloc(pts, n)
    dvz.point_position(pts, 0, n, pos3, 0)
    dvz.point_color(pts, 0, n, col, 0)
    dvz.point_size(pts, 0, n, sizes, 0)
    dvz.panel_visual(panel, pts, 0)

    dvz.scene_run(scene, app, 2)
    dvz.app_screenshot(app, dvz.figure_id(fig), out_path.encode("utf-8"))
    dvz.scene_destroy(scene)
    dvz.app_destroy(app)

    return np.asarray(Image.open(out_path).convert("RGBA"))


def run_interactive_window(n: int = NUM_BODIES) -> None:
    """Launch a live 60+ FPS native Vulkan Datoviz window with Dear ImGui controls."""
    init_pm, init_vg = create_colliding_galaxies(n)
    sim = vksp.BarnesHut3D(init_pm, init_vg, G=0.025, damping=0.9999)

    app = dvz.app(dvz.APP_FLAGS_NONE)
    batch = dvz.app_batch(app)
    scene = dvz.scene(batch)
    fig = dvz.figure(scene, 1100, 720, dvz.CANVAS_FLAGS_IMGUI | dvz.CANVAS_FLAGS_FPS)
    panel = dvz.panel_default(fig)
    dvz.panel_arcball(panel, 0)
    cam = dvz.panel_camera(panel, 0)
    dvz.camera_position(cam, dvz.vec3(0.4, 1.2, 2.2))
    dvz.camera_lookat(cam, dvz.vec3(0.0, 0.0, 0.0))

    pos3 = np.ascontiguousarray(init_pm[:, :3] * WORLD_SCALE, dtype=np.float32)
    col = compute_star_colors(init_vg, init_pm[:, 3])
    sizes = np.where(init_pm[:, 3] > 100.0, 18.0, 3.2).astype(np.float32)

    pts = dvz.point(batch, 0)
    dvz.point_alloc(pts, n)
    dvz.point_position(pts, 0, n, pos3, 0)
    dvz.point_color(pts, 0, n, col, 0)
    dvz.point_size(pts, 0, n, sizes, 0)
    dvz.panel_visual(panel, pts, 0)

    c_running = ctypes.c_bool(True)
    c_G = ctypes.c_float(0.025)
    c_steps = ctypes.c_float(2.0)

    @dvz.timer
    def on_timer(app_, fig_id, ev):
        if not c_running.value:
            return
        sim.G = float(c_G.value)
        sim.step(max(1, int(round(c_steps.value))))
        pm = sim.pos_mass.to_numpy()
        vg = sim.vel_gal.to_numpy()
        p_new = np.ascontiguousarray(pm[:, :3] * WORLD_SCALE, dtype=np.float32)
        c_new = compute_star_colors(vg, pm[:, 3])
        dvz.point_position(pts, 0, n, p_new, 0)
        dvz.point_color(pts, 0, n, c_new, 0)
        dvz.visual_update(pts)

    @dvz.gui
    def on_gui(app_, fig_id, ev):
        dvz.gui_pos(dvz.vec2(16, 16), dvz.vec2(0, 0))
        dvz.gui_size(dvz.vec2(260, 165))
        dvz.gui_begin(b"vkdata + Datoviz N-Body", 0)
        dvz.gui_checkbox(b"Running", ctypes.byref(c_running))
        dvz.gui_slider(b"Gravity G", 0.005, 0.12, ctypes.byref(c_G))
        dvz.gui_slider(b"Steps/Frame", 1.0, 6.0, ctypes.byref(c_steps))
        if dvz.gui_button(b"Reset Galaxies", 140.0, 26.0):
            sim.reset(init_pm, init_vg)
        dvz.gui_end()

    dvz.app_timer(app, 0.0, 0.016, 0)
    dvz.app_ontimer(app, on_timer, None)
    dvz.app_gui(app, dvz.figure_id(fig), on_gui, None)
    dvz.scene_run(scene, app, 0)
    dvz.scene_destroy(scene)
    dvz.app_destroy(app)


def main():
    parser = argparse.ArgumentParser(description="vkspatial.BarnesHut3D + Datoviz 16,384-Body Galaxies")
    parser.add_argument("--offscreen", type=str, default="", help="Render offscreen PNG to path and exit")
    parser.add_argument("--steps", type=int, default=140, help="Simulation steps before offscreen snapshot")
    args = parser.parse_args()

    if args.offscreen:
        pm, vg = create_colliding_galaxies(NUM_BODIES)
        sim = vksp.BarnesHut3D(pm, vg, G=0.025, damping=0.9999)
        if args.steps > 4:
            sim.step(args.steps - 4)
        prev = sim.pos_mass.to_numpy()[:, :3].copy()
        sim.step(4)
        rgba = render_datoviz_snapshot(
            sim.pos_mass.to_numpy(), sim.vel_gal.to_numpy(), prev, out_path=args.offscreen
        )
        print(f"Saved Datoviz Vulkan snapshot ({rgba.shape[1]}x{rgba.shape[0]}) to {args.offscreen}")
    else:
        run_interactive_window(NUM_BODIES)


if __name__ == "__main__":
    main()
