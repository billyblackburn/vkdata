"""Native Vulkan Hardware-Rasterized 16,384-Body Colliding Galaxies (vkspatial.BarnesHut3D + Datoviz).

Combines:
- `vkspatial.BarnesHut3D`: 6-level 3D Morton Octree + elastic collisions on Vulkan compute (~1.2 ms/step)
- `datoviz`: Native Vulkan hardware-rasterized 3D stars + multi-layer volumetric HDR bloom halos + Arcball + Dear ImGui HUD
- Distinct galactic morphologies: Barred 4-Arm Spiral vs. Cartwheel Ring Galaxy (plus 3-Way Merger & Polar Ring presets!)

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
G_DEFAULT = 0.025


def _tilt_and_place(px, py, pz, vx, vy, vz, center, bulk_vel, tilt_x, tilt_z=0.0):
    cx, sx = math.cos(tilt_x), math.sin(tilt_x)
    py, pz = cx * py - sx * pz, sx * py + cx * pz
    vy, vz = cx * vy - sx * vz, sx * vy + cx * vz

    if abs(tilt_z) > 1e-6:
        cz, sz = math.cos(tilt_z), math.sin(tilt_z)
        px, py = cz * px - sz * py, sz * px + cz * py
        vx, vy = cz * vx - sz * vy, sz * vx + cz * vy

    pos = np.stack([px, py, pz], axis=-1) + np.asarray(center, dtype=np.float32)
    vel = np.stack([vx, vy, vz], axis=-1) + np.asarray(bulk_vel, dtype=np.float32)
    return pos.astype(np.float32), vel.astype(np.float32)


def _make_barred_spiral(rng, count: int, center, bulk_vel, tilt_x: float, tilt_z: float, gal_id: float, core_mass: float = 1900.0):
    """4-arm barred grand-design spiral galaxy (elongated central bar + 4 sweeping logarithmic arms)."""
    n_bar = count // 4
    n_arm = count - n_bar

    # Central elongated stellar bar
    bx = rng.normal(0.0, 4.8, n_bar).astype(np.float32)
    bz = rng.normal(0.0, 1.35, n_bar).astype(np.float32)
    by = rng.normal(0.0, 0.55, n_bar).astype(np.float32)
    r_bar = np.sqrt(bx * bx + bz * bz) + 0.5
    ang_bar = np.arctan2(bz, bx)

    # 4 logarithmic spiral arms starting from the bar tips
    u = rng.uniform(0.0, 1.0, n_arm).astype(np.float32)
    r_arm = (4.2 + 16.5 * (u ** 0.80)).astype(np.float32)
    arm_id = rng.integers(0, 4, n_arm).astype(np.float32) * (0.5 * np.pi)
    ang_arm = (arm_id + 0.28 * r_arm + rng.normal(0.0, 0.24, n_arm)).astype(np.float32)
    ax = r_arm * np.cos(ang_arm)
    az = r_arm * np.sin(ang_arm)
    ay = rng.normal(0.0, 0.30, n_arm).astype(np.float32)

    px = np.concatenate([bx, ax])
    py = np.concatenate([by, ay])
    pz = np.concatenate([bz, az])
    radii = np.concatenate([r_bar, r_arm])
    angles = np.concatenate([ang_bar, ang_arm])

    enclosed = core_mass + (radii / 20.5) * (count * 0.45)
    v_orb = np.sqrt(G_DEFAULT * enclosed / np.maximum(radii, 0.7))
    vx = -v_orb * np.sin(angles)
    vy = np.zeros(count, dtype=np.float32)
    vz = v_orb * np.cos(angles)

    pos, vel = _tilt_and_place(px, py, pz, vx, vy, vz, center, bulk_vel, tilt_x, tilt_z)
    mass = rng.uniform(0.25, 0.90, count).astype(np.float32)
    pos[0], vel[0], mass[0] = center, bulk_vel, core_mass
    return pos, vel, mass, np.full(count, gal_id, dtype=np.float32)


def _make_cartwheel_ring(rng, count: int, center, bulk_vel, tilt_x: float, tilt_z: float, gal_id: float, core_mass: float = 1800.0):
    """Cartwheel Ring Galaxy (luminous outer star-forming ring + 6 radial spokes + bright compact bulge)."""
    n_bulge = count // 5
    n_spokes = count // 4
    n_ring = count - n_bulge - n_spokes

    # Compact spherical inner bulge
    r_b = np.abs(rng.normal(0.0, 2.4, n_bulge)).astype(np.float32) + 0.4
    th_b = rng.uniform(0.0, 2.0 * np.pi, n_bulge).astype(np.float32)
    ph_b = rng.normal(0.0, 0.45, n_bulge).astype(np.float32)
    bx, by, bz = r_b * np.cos(th_b) * np.cos(ph_b), r_b * np.sin(ph_b), r_b * np.sin(th_b) * np.cos(ph_b)

    # 6 curved radial spokes bridging bulge to outer ring
    r_s = rng.uniform(2.5, 14.5, n_spokes).astype(np.float32)
    spoke = rng.integers(0, 6, n_spokes).astype(np.float32) * (np.pi / 3.0)
    th_s = (spoke + 0.08 * r_s + rng.normal(0.0, 0.09, n_spokes)).astype(np.float32)
    sx, sy, sz = r_s * np.cos(th_s), rng.normal(0.0, 0.25, n_spokes).astype(np.float32), r_s * np.sin(th_s)

    # Dense luminous outer ring (r ~ 14.5 +/- 1.8)
    r_r = np.clip(rng.normal(14.8, 1.45, n_ring), 10.5, 19.5).astype(np.float32)
    th_r = rng.uniform(0.0, 2.0 * np.pi, n_ring).astype(np.float32)
    rx, ry, rz = r_r * np.cos(th_r), rng.normal(0.0, 0.35, n_ring).astype(np.float32), r_r * np.sin(th_r)

    px = np.concatenate([bx, sx, rx])
    py = np.concatenate([by, sy, ry])
    pz = np.concatenate([bz, sz, rz])
    radii = np.concatenate([r_b, r_s, r_r])
    angles = np.concatenate([th_b, th_s, th_r])

    enclosed = core_mass + (radii / 19.5) * (count * 0.45)
    v_orb = np.sqrt(G_DEFAULT * enclosed / np.maximum(radii, 0.7))
    vx = -v_orb * np.sin(angles)
    vy = np.zeros(count, dtype=np.float32)
    vz = v_orb * np.cos(angles)

    pos, vel = _tilt_and_place(px, py, pz, vx, vy, vz, center, bulk_vel, tilt_x, tilt_z)
    mass = rng.uniform(0.25, 0.90, count).astype(np.float32)
    pos[0], vel[0], mass[0] = center, bulk_vel, core_mass
    return pos, vel, mass, np.full(count, gal_id, dtype=np.float32)


def _make_elliptical_glob(rng, count: int, center, bulk_vel, gal_id: float, core_mass: float = 1500.0):
    """3D triaxial elliptical galaxy with swirling internal angular momentum."""
    u = rng.uniform(0.0, 1.0, count).astype(np.float32)
    r = (0.8 + 11.5 * (u ** 1.2)).astype(np.float32)
    th = rng.uniform(0.0, 2.0 * np.pi, count).astype(np.float32)
    cos_p = rng.uniform(-1.0, 1.0, count).astype(np.float32)
    sin_p = np.sqrt(np.maximum(1.0 - cos_p * cos_p, 0.0))
    px = r * sin_p * np.cos(th) * 1.25
    py = r * cos_p * 0.65
    pz = r * sin_p * np.sin(th) * 0.95
    enclosed = core_mass + (r / 12.5) * (count * 0.4)
    v_orb = 0.72 * np.sqrt(G_DEFAULT * enclosed / np.maximum(r, 0.8))
    vx = -v_orb * np.sin(th) + rng.normal(0.0, 0.35, count).astype(np.float32)
    vy = rng.normal(0.0, 0.35, count).astype(np.float32)
    vz = v_orb * np.cos(th) + rng.normal(0.0, 0.35, count).astype(np.float32)
    pos, vel = _tilt_and_place(px, py, pz, vx, vy, vz, center, bulk_vel, 0.3, -0.2)
    mass = rng.uniform(0.25, 0.90, count).astype(np.float32)
    pos[0], vel[0], mass[0] = center, bulk_vel, core_mass
    return pos, vel, mass, np.full(count, gal_id, dtype=np.float32)


def create_colliding_galaxies(n: int = NUM_BODIES, seed: int = 42, preset: str = "barred_vs_ring"):
    """Initialize colliding galaxies with distinct 3D morphologies.

    Presets:
      - ``"barred_vs_ring"``: 4-Arm Barred Spiral (Cyan) vs. Cartwheel Ring Galaxy (Rose-Gold)
      - ``"triple_merger"``: 3-Way Clash (Barred Spiral + Cartwheel Ring + Triaxial Elliptical)
      - ``"polar_ring"``: Perpendicular Polar-Ring collision (two orthogonal Cartwheel Rings)
    """
    rng = np.random.default_rng(seed)
    pos_mass = np.zeros((n, 4), dtype=np.float32)
    vel_gal = np.zeros((n, 4), dtype=np.float32)

    if preset == "triple_merger":
        c1 = n // 3
        c2 = n // 3
        c3 = n - c1 - c2
        parts = [
            _make_barred_spiral(rng, c1, [-17.5,  4.0, -9.0], [ 1.20, -0.25,  0.65],  0.35,  0.20, 0.0, 1500.0),
            _make_cartwheel_ring(rng, c2, [ 17.5, -3.0, -5.0], [-1.25,  0.20,  0.40], -0.65,  0.45, 1.0, 1450.0),
            _make_elliptical_glob(rng, c3, [  0.0, -2.5, 19.0], [ 0.05,  0.10, -1.35], 2.0, 1400.0),
        ]
    elif preset == "polar_ring":
        half = n // 2
        parts = [
            _make_cartwheel_ring(rng, half,     [-15.5,  2.5, -5.0], [ 1.15, -0.18,  0.45],  0.15, 0.00, 0.0, 1850.0),
            _make_cartwheel_ring(rng, n - half, [ 15.5, -2.5,  5.0], [-1.15,  0.18, -0.45],  1.45, 0.35, 1.0, 1850.0),
        ]
    else:  # "barred_vs_ring"
        half = n // 2
        parts = [
            _make_barred_spiral(rng, half,     [-17.0,  3.5, -5.5], [ 1.25, -0.22,  0.50],  0.32,  0.15, 0.0, 1900.0),
            _make_cartwheel_ring(rng, n - half, [ 17.0, -3.5,  5.5], [-1.25,  0.22, -0.50], -0.78,  0.42, 1.0, 1800.0),
        ]

    offset = 0
    for pos, vel, mass, gid in parts:
        k = pos.shape[0]
        pos_mass[offset : offset + k, 0:3] = pos
        pos_mass[offset : offset + k, 3] = mass
        vel_gal[offset : offset + k, 0:3] = vel
        vel_gal[offset : offset + k, 3] = gid
        offset += k

    return pos_mass, vel_gal


def compute_star_colors(vel_gal_np: np.ndarray, masses: np.ndarray, alpha_val: int = 195) -> np.ndarray:
    """Compute vivid RGBA uint8 colors for up to 3 galaxy types + white-hot collision flashes."""
    n = vel_gal_np.shape[0]
    gal = np.floor(vel_gal_np[:, 3])
    flash = np.clip(vel_gal_np[:, 3] - gal, 0.0, 1.0)
    speed = np.linalg.norm(vel_gal_np[:, :3], axis=1)
    t = np.clip(speed / 5.5, 0.0, 1.0)

    # Galaxy 0: Electric Cyan-Blue | Galaxy 1: Rose-Magenta -> Gold | Galaxy 2: Emerald-Violet
    is_g0 = gal < 0.5
    is_g1 = (gal >= 0.5) & (gal < 1.5)
    r = np.where(is_g0, 0.15 + 0.70 * t, np.where(is_g1, 1.00, 0.30 + 0.55 * t))
    g = np.where(is_g0, 0.62 + 0.35 * t, np.where(is_g1, 0.24 + 0.68 * t, 0.92 - 0.25 * t))
    b = np.where(is_g0, 1.00, np.where(is_g1, 0.55 - 0.30 * t, 0.72 + 0.28 * t))

    boost = flash * 0.95
    r = np.clip(r + boost, 0.0, 1.0)
    g = np.clip(g + boost, 0.0, 1.0)
    b = np.clip(b + boost, 0.0, 1.0)

    is_core = masses > 100.0
    r[is_core], g[is_core], b[is_core] = 1.0, 0.96, 0.85

    rgba = np.empty((n, 4), dtype=np.uint8)
    rgba[:, 0] = (r * 255.0).astype(np.uint8)
    rgba[:, 1] = (g * 255.0).astype(np.uint8)
    rgba[:, 2] = (b * 255.0).astype(np.uint8)
    rgba[:, 3] = int(np.clip(alpha_val, 0, 255))
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
    col = compute_star_colors(vel_gal_np, pos_mass_np[:, 3], alpha_val=175)
    sizes = np.where(pos_mass_np[:, 3] > 100.0, 14.0, 2.0).astype(np.float32)

    app = dvz.app(dvz.APP_FLAGS_OFFSCREEN)
    batch = dvz.app_batch(app)
    scene = dvz.scene(batch)
    fig = dvz.figure(scene, width, height, 0)
    panel = dvz.panel_default(fig)
    dvz.panel_arcball(panel, 0)
    cam = dvz.panel_camera(panel, 0)
    dvz.camera_zrange(cam, 0.01, 50.0)

    cx = float(dist * math.cos(pitch) * math.sin(yaw))
    cy = float(dist * math.sin(pitch))
    cz = float(dist * math.cos(pitch) * math.cos(yaw))
    dvz.camera_position(cam, dvz.vec3(cx, cy, cz))
    dvz.camera_lookat(cam, dvz.vec3(0.0, 0.0, 0.0))

    if prev_pos_np is not None:
        p0 = np.ascontiguousarray(prev_pos_np * WORLD_SCALE, dtype=np.float32)
        seg = dvz.segment(batch, 0)
        dvz.segment_alloc(seg, n)
        dvz.segment_position(seg, 0, n, p0, pos3, 0)
        seg_col = col.copy()
        seg_col[:, 3] = 95
        dvz.segment_color(seg, 0, n, seg_col, 0)
        lw = np.full(n, 1.4, dtype=np.float32)
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


def run_interactive_window(n: int = NUM_BODIES, preset: str = "barred_vs_ring") -> None:
    """Launch a live 60+ FPS native Vulkan Datoviz window with multi-layer HDR bloom halos & Dear ImGui HUD."""
    state = {"preset": preset}
    init_pm, init_vg = create_colliding_galaxies(n, preset=preset)
    sim = vksp.BarnesHut3D(init_pm, init_vg, G=G_DEFAULT, damping=0.9999)

    app = dvz.app(dvz.APP_FLAGS_NONE)
    batch = dvz.app_batch(app)
    scene = dvz.scene(batch)
    fig = dvz.figure(scene, 1140, 740, dvz.CANVAS_FLAGS_IMGUI | dvz.CANVAS_FLAGS_FPS)
    panel = dvz.panel_default(fig)
    arc = dvz.panel_arcball(panel, 0)
    cam = dvz.panel_camera(panel, 0)
    dvz.camera_zrange(cam, 0.01, 50.0)

    cam_dir = np.array([0.35, 1.15, 2.05], dtype=np.float32)
    init_dist = float(np.linalg.norm(cam_dir))
    cam_dir = cam_dir / init_dist
    c_dist = ctypes.c_float(init_dist)

    def _apply_zoom():
        d = float(np.clip(c_dist.value, 0.12, 12.0))
        c_dist.value = d
        dvz.camera_position(cam, dvz.vec3(float(cam_dir[0] * d), float(cam_dir[1] * d), float(cam_dir[2] * d)))
        dvz.camera_lookat(cam, dvz.vec3(0.0, 0.0, 0.0))
        dvz.panel_update(panel)

    _apply_zoom()

    # Subsample 1/4 of stars for the wide nebular bloom halo (fast + silky smooth!)
    bloom_idx = np.arange(0, n, 4, dtype=np.int32)
    # Always include supermassive cores in bloom layer
    core_indices = np.nonzero(init_pm[:, 3] > 100.0)[0]
    for ci in core_indices:
        bloom_idx[0] = ci
    nb = bloom_idx.size

    pos3 = np.ascontiguousarray(init_pm[:, :3] * WORLD_SCALE, dtype=np.float32)
    col_core = compute_star_colors(init_vg, init_pm[:, 3], alpha_val=210)
    sizes_core = np.where(init_pm[:, 3] > 100.0, 14.0, 2.6).astype(np.float32)

    # Layer 1 (back): Wide nebular HDR bloom halo
    halo_wide = dvz.point(batch, 0)
    dvz.point_alloc(halo_wide, nb)
    dvz.visual_depth(halo_wide, dvz.DEPTH_TEST_DISABLE)
    dvz.panel_visual(panel, halo_wide, 0)

    # Layer 2 (mid): Tight bright HDR bloom halo
    halo_tight = dvz.point(batch, 0)
    dvz.point_alloc(halo_tight, n)
    dvz.visual_depth(halo_tight, dvz.DEPTH_TEST_DISABLE)
    dvz.panel_visual(panel, halo_tight, 0)

    # Layer 3 (front): Crisp star cores & white-hot collision sparks
    pts = dvz.point(batch, 0)
    dvz.point_alloc(pts, n)
    dvz.panel_visual(panel, pts, 0)

    c_running = ctypes.c_bool(True)
    c_G = ctypes.c_float(G_DEFAULT)
    c_steps = ctypes.c_float(2.0)
    c_bloom = ctypes.c_float(1.15)

    def _upload_visuals(pm: np.ndarray, vg: np.ndarray):
        b_amt = float(c_bloom.value)
        p_all = np.ascontiguousarray(pm[:, :3] * WORLD_SCALE, dtype=np.float32)
        masses = pm[:, 3]
        is_core = masses > 100.0

        # Crisp stars
        c_pts = compute_star_colors(vg, masses, alpha_val=215)
        s_pts = np.where(is_core, 13.0, 2.5).astype(np.float32)
        dvz.point_position(pts, 0, n, p_all, 0)
        dvz.point_color(pts, 0, n, c_pts, 0)
        dvz.point_size(pts, 0, n, s_pts, 0)
        dvz.visual_update(pts)

        # Tight HDR bloom halo
        c_tight = c_pts.copy()
        a_t = int(np.clip(28.0 * b_amt, 0, 95))
        c_tight[:, 3] = a_t
        c_tight[is_core, 3] = int(np.clip(110.0 * b_amt, 0, 210))
        s_tight = np.where(is_core, 42.0 * max(0.3, b_amt), 9.5 * max(0.2, b_amt)).astype(np.float32)
        dvz.point_position(halo_tight, 0, n, p_all, 0)
        dvz.point_color(halo_tight, 0, n, c_tight, 0)
        dvz.point_size(halo_tight, 0, n, s_tight, 0)
        dvz.visual_update(halo_tight)

        # Wide nebular HDR bloom halo (subsampled + supermassive cores)
        p_w = np.ascontiguousarray(p_all[bloom_idx], dtype=np.float32)
        c_w = c_pts[bloom_idx].copy()
        w_core = is_core[bloom_idx]
        c_w[:, 3] = int(np.clip(12.0 * b_amt, 0, 48))
        c_w[w_core, 3] = int(np.clip(65.0 * b_amt, 0, 160))
        s_w = np.where(w_core, 96.0 * max(0.3, b_amt), 26.0 * max(0.2, b_amt)).astype(np.float32)
        dvz.point_position(halo_wide, 0, nb, p_w, 0)
        dvz.point_color(halo_wide, 0, nb, c_w, 0)
        dvz.point_size(halo_wide, 0, nb, s_w, 0)
        dvz.visual_update(halo_wide)

    _upload_visuals(init_pm, init_vg)

    def _switch_preset(new_preset: str):
        nonlocal init_pm, init_vg
        state["preset"] = new_preset
        init_pm, init_vg = create_colliding_galaxies(n, preset=new_preset)
        sim.reset(init_pm, init_vg)
        _upload_visuals(init_pm, init_vg)

    @dvz.mouse
    def on_mouse(app_, fig_id, ev):
        if ev.type == dvz.MOUSE_EVENT_WHEEL:
            dy = float(ev.content.w.dir[1])
            if abs(dy) > 1e-6:
                c_dist.value = float(np.clip(c_dist.value * (0.82 ** dy), 0.12, 12.0))
                _apply_zoom()

    @dvz.timer
    def on_timer(app_, fig_id, ev):
        if not c_running.value:
            return
        sim.G = float(c_G.value)
        sim.step(max(1, int(round(c_steps.value))))
        _upload_visuals(sim.pos_mass.to_numpy(), sim.vel_gal.to_numpy())

    @dvz.gui
    def on_gui(app_, fig_id, ev):
        dvz.gui_pos(dvz.vec2(16, 16), dvz.vec2(0, 0))
        dvz.gui_size(dvz.vec2(290, 315))
        dvz.gui_begin(b"vkdata + Datoviz HDR N-Body", 0)
        dvz.gui_checkbox(b"Running", ctypes.byref(c_running))
        dvz.gui_slider(b"Gravity G", 0.005, 0.12, ctypes.byref(c_G))
        dvz.gui_slider(b"Steps/Frame", 1.0, 6.0, ctypes.byref(c_steps))
        prev_b = c_bloom.value
        dvz.gui_slider(b"HDR Bloom", 0.0, 2.0, ctypes.byref(c_bloom))
        if abs(c_bloom.value - prev_b) > 1e-3 and not c_running.value:
            _upload_visuals(sim.pos_mass.to_numpy(), sim.vel_gal.to_numpy())
        prev_d = c_dist.value
        dvz.gui_slider(b"Zoom Dist", 0.12, 8.0, ctypes.byref(c_dist))
        if abs(c_dist.value - prev_d) > 1e-4:
            _apply_zoom()
        if dvz.gui_button(b"Barred Spiral x Ring", 250.0, 24.0):
            _switch_preset("barred_vs_ring")
        if dvz.gui_button(b"3-Way Galaxy Merger", 250.0, 24.0):
            _switch_preset("triple_merger")
        if dvz.gui_button(b"Polar Ring Collision", 250.0, 24.0):
            _switch_preset("polar_ring")
        if dvz.gui_button(b"Reset View", 250.0, 24.0):
            dvz.arcball_reset(arc)
            c_dist.value = init_dist
            _apply_zoom()
        dvz.gui_end()

    dvz.app_timer(app, 0.0, 0.016, 0)
    dvz.app_ontimer(app, on_timer, None)
    dvz.app_onmouse(app, on_mouse, None)
    dvz.app_gui(app, dvz.figure_id(fig), on_gui, None)
    dvz.scene_run(scene, app, 0)
    dvz.scene_destroy(scene)
    dvz.app_destroy(app)


def main():
    parser = argparse.ArgumentParser(description="vkspatial.BarnesHut3D + Datoviz 16,384-Body Galaxies")
    parser.add_argument("--offscreen", type=str, default="", help="Render offscreen PNG to path and exit")
    parser.add_argument("--steps", type=int, default=140, help="Simulation steps before offscreen snapshot")
    parser.add_argument("--preset", type=str, default="barred_vs_ring", choices=["barred_vs_ring", "triple_merger", "polar_ring"])
    args = parser.parse_args()

    if args.offscreen:
        pm, vg = create_colliding_galaxies(NUM_BODIES, preset=args.preset)
        sim = vksp.BarnesHut3D(pm, vg, G=G_DEFAULT, damping=0.9999)
        if args.steps > 4:
            sim.step(args.steps - 4)
        prev = sim.pos_mass.to_numpy()[:, :3].copy()
        sim.step(4)
        rgba = render_datoviz_snapshot(
            sim.pos_mass.to_numpy(), sim.vel_gal.to_numpy(), prev, out_path=args.offscreen
        )
        print(f"Saved Datoviz Vulkan snapshot ({rgba.shape[1]}x{rgba.shape[0]}) to {args.offscreen}")
    else:
        run_interactive_window(NUM_BODIES, preset=args.preset)


if __name__ == "__main__":
    main()
