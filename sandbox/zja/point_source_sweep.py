"""
Visualize off-axis companion injection driven entirely by default_param.yml.

Purpose
-------
Generate corgisim GITL frames (host-only / companion-only / combined, all in
log scale) WITHOUT running the Jacobian / probe / EETC machinery in
run_corgisim_nulling_gitl.py. All model settings (mode, dark_hole, DM start
maps, corgi_overrides incl. the companion) are read from default_param.yml so
this always reflects whatever the real loop would actually use -- nothing is
hardcoded here except the sweep range itself.

This targets the corgisim (corgihowfsc) backend specifically: off-axis
companions don't exist for the compact model (cgi-howfsc), so this script
always reads params['models']['corgihowfsc'], regardless of the yaml's
top-level active_model.

Modes
-----
1. Default (no --sweep): uses the single companion position currently set in
   default_param.yml's models.corgihowfsc.corgi_overrides.point_sources, and
   saves a 3-panel figure (host / companion / combined, log scale).

2. --sweep: sweeps the companion's x position from --x-start to --x-stop in
   --x-step increments (mas), holding vmag and y fixed at the yaml's values.
   For each position, saves a log-scale combined-image PNG (fixed color scale
   across the whole sweep) and computes the mean contrast in the real
   dark-hole control mask (cfg.sl_list[lind].dh.e). Produces a .gif from the
   PNGs and a mean-contrast-vs-separation plot.

Run
---
    python sandbox/zja/point_source_sweep.py
    python sandbox/zja/point_source_sweep.py --sweep
    python sandbox/zja/point_source_sweep.py --sweep --x-stop 2000 --x-step 500  # quick check
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from PIL import Image
from matplotlib.animation import FuncAnimation, PillowWriter
from datetime import datetime
from pathlib import Path

import roman_preflight_proper
roman_preflight_proper.copy_here()

import corgihowfsc
from howfsc.model.mode import CoronagraphMode
from howfsc.util.loadyaml import loadyaml
from howfsc.util.insertinto import insertinto

from corgihowfsc.utils.howfsc_initialization import get_args, load_files
from corgihowfsc.utils.corgisim_manager import CorgisimManager
from corgihowfsc.utils.contrast_nomalization import CorgiNormalizationOnAxis
from corgihowfsc.utils.corgisim_utils import calculate_mas_per_lamD

HOWFSCPATH = os.path.dirname(os.path.abspath(corgihowfsc.__file__))
DEFAULT_PARAM_FILE = os.path.join(HOWFSCPATH, 'scripts', 'default_param.yml')
# DEFAULT_OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = "C:/Users/zahmed1/corgiloop_data/off_axis_star_testing"

def fov_range_lamd(cor_mapped):
    """Mirror corgisim/instrument.py's inject_point_sources FOV_range lookup
    (lambda/D), so the reference lines match whichever mode is actually running."""
    if 'hlc' in cor_mapped:
        return 3.0, 9.7
    elif 'spec' in cor_mapped:
        return 3.0, 9.1
    else:
        return 5.9, 20.1


def build_inputs(param_file):
    """Load cfg / hconf / DM start maps / corgi_overrides straight from
    default_param.yml, exactly as run_corgisim_nulling_gitl.py does, minus
    the Jacobian / probe / EETC / cstrat machinery that isn't needed just to
    generate an image."""
    params = loadyaml(param_file)
    model_cfg = params['models']['corgihowfsc']

    mode = params['sim_settings']['mode']
    dark_hole = params['sim_settings']['dark_hole']
    output_dim = params['crop']['nrow']

    path_overrides = {
        k: v for k, v in params.get('path_overrides', {}).items() if v is not None
    }

    args = get_args(
        mode=mode,
        dark_hole=dark_hole,
        probe_shape=params['sim_settings']['probe_shape'],
        jacpath=None,  # no Jacobian needed for image generation
        dmstartmap_filenames=model_cfg['dmstartmap_filenames'],
        path_overrides=path_overrides,
    )

    (modelpath, cfgfile, jacfile, cstratfile, probefiles,
     hconffile, n2clistfiles, dmstartmaps) = load_files(args, HOWFSCPATH)

    cfg = CoronagraphMode(cfgfile)
    hconf = loadyaml(hconffile, custom_exception=TypeError)
    dm1v, dm2v = dmstartmaps[0], dmstartmaps[1]

    base_overrides = dict(model_cfg.get('corgi_overrides', {}))
    base_overrides['output_dim'] = output_dim

    return cfg, hconf, mode, dm1v, dm2v, base_overrides


def make_manager(cfg, hconf, mode, overrides):
    # CorgisimManager doesn't use cstrat for anything -- only cfg.sl_list --
    # so it's fine to skip building a real ControlStrategy here.
    return CorgisimManager(cfg, None, hconf, cor=mode, corgi_overrides=overrides)


def without_point_sources(overrides):
    return {k: v for k, v in overrides.items() if k != 'point_sources'}


def clip_positive(img, floor_frac=1e-6):
    """Clip to a small positive floor (relative to the image's own peak) so the
    image is safe to render with LogNorm -- exact zeros/negative numerical
    noise (e.g. from combined - host subtraction) would otherwise break it."""
    img = np.asarray(img, dtype=float)
    peak = np.nanmax(img)
    floor = max(peak * floor_frac, 1e-30)
    return np.clip(img, floor, None)


def shared_log_range(imgs, floor_frac=1e-6):
    """Common (vmin, vmax) for LogNorm across multiple images, so they share
    one log color scale instead of each auto-scaling to its own peak."""
    vmax = max(np.nanmax(img) for img in imgs)
    # vmin = max(vmax * floor_frac, 1e-30)
    vmin = min(np.nanmin(img) for img in imgs)
    return vmin, vmax


def build_normalizer(cfg, hconf, mode, host_overrides, dm1v, dm2v, lind, exptime):
    """Peak flux (unocculted host star) for NI normalization -- reuse the same
    convention the real loop uses for corgisim (CorgiNormalizationOnAxis)."""
    norm = CorgiNormalizationOnAxis(cfg, None, hconf, cor=mode,
                                     corgi_overrides=host_overrides, exptime_norm=exptime)
    _, peakflux = norm.calc_flux_rate(None, hconf, lind, dm1v, dm2v)
    return norm, peakflux


def make_three_panel(host_img, companion_img, combined_img, out_path, title, ppl):
    imgs = [host_img, companion_img, combined_img]
    vmin, vmax = shared_log_range(imgs)

    ny, nx = host_img.shape
    extent = [-(nx // 2) / ppl, (nx // 2) / ppl, -(ny // 2) / ppl, (ny // 2) / ppl]  # lambda/D

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for ax, img, panel_title in zip(
        axes,
        imgs,
        ['Host star only', 'Off-axis companion only', 'Combined'],
    ):
        im = ax.imshow(np.clip(img, vmin, None), origin='lower', cmap='inferno',
                        norm=LogNorm(vmin=vmin, vmax=vmax), extent=extent)
        ax.set_title(panel_title, fontsize=16)
        ax.set_xlabel(r'x [$\lambda/D$]', fontsize=16)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        if panel_title == 'Host star only':
            ax.set_ylabel(r"y [$\lambda/D$]", fontsize=16)
        # if panel_title == 'Combined':
        #     fig.colorbar(im, ax=ax, fraction=0.046, label='Normalized intensity').ax.tick_params(labelsize=14)   
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, label='Normalized intensity')
    cbar.ax.tick_params(labelsize=14)  
    cbar.set_label("Normalized Intensity", fontsize=14)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f'Saved {out_path}')


def save_log_frame(img, path, out_dir, frame_name, vmin, vmax, title):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(clip_positive(img), origin='lower', cmap='inferno',
                    norm=LogNorm(vmin=vmin, vmax=vmax))
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, label='Normalized intensity')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)

    data_name = frame_name + '.npy'
    np.save(os.path.join(out_dir,data_name), clip_positive(img))


def make_gif(frame_paths, out_path, duration_ms=800):
    frames = [Image.open(p).convert('RGB') for p in frame_paths]
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                    duration=duration_ms, loop=0)
    print(f'Saved {out_path}')


def plot_contrast(x_mas, c_combined, c_companion, host_contrast, mas_per_lamd,
                   iwa_lamd, owa_lamd, mode_label, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    x_lamd = x_mas / mas_per_lamd
    ax.semilogy(x_lamd, c_combined, 'o-', label='host + companion (dark-hole mean)')
    ax.semilogy(x_lamd, c_companion, 's--', label='companion contribution only')
    ax.axhline(host_contrast, color='gray', linestyle=':', label='host-only baseline')

    ax.axvline(iwa_lamd, color='orange', linestyle='--', alpha=0.6,
               label=r'{mode_label} IWA ({iwa_lamd} $\lambda/D$)')
    ax.axvline(owa_lamd, color='red', linestyle='--', alpha=0.6,
               label=r'{mode_label} OWA ({owa_lamd} $\lambda/D$)')

    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlabel(r'Binary Separation [$\lambda/D$]', fontsize=16)
    ax.set_ylabel('Normalized Intensity', fontsize=16)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved {out_path}')


def run_single(cfg, hconf, mode, dm1v, dm2v, base_overrides, lind, exptime, out_dir):
    point_sources = base_overrides.get('point_sources')
    if not point_sources:
        raise ValueError(
            "No companion configured in default_param.yml's "
            "models.corgihowfsc.corgi_overrides.point_sources -- nothing to plot. "
            "Set one, or use --sweep to sweep positions explicitly."
        )

    host_overrides = without_point_sources(base_overrides)
    mgr_host = make_manager(cfg, hconf, mode, host_overrides)
    host_img = mgr_host.generate_host_star_psf(dm1v, dm2v, lind=lind, exptime=exptime)

    mgr_full = make_manager(cfg, hconf, mode, base_overrides)
    combined_img = mgr_full.generate_host_star_psf(dm1v, dm2v, lind=lind, exptime=exptime)

    companion_img = combined_img - host_img

    norm, peakflux = build_normalizer(cfg, hconf, mode, host_overrides, dm1v, dm2v, lind, exptime)
    host_ni = norm.normalize(host_img, peakflux, exptime)
    combined_ni = norm.normalize(combined_img, peakflux, exptime)
    companion_ni = norm.normalize(companion_img, peakflux, exptime)

    src = point_sources[0]
    title = (f"{mode}, companion vmag={src['vmag']} at "
             f"(dx={src['position_x_mas']}, dy={src['position_y_mas']}) mas")
    out_path = os.path.join(out_dir, 'point_source_three_panel.png')

    ppl_lam0 = 3.14 # hardcoding for now but should make this better

    make_three_panel(host_ni, companion_ni, combined_ni, out_path, title, ppl_lam0)


def run_sweep(cfg, hconf, mode, dm1v, dm2v, base_overrides, lind, exptime, out_dir,
              x_start, x_stop, x_step):
    point_sources = base_overrides.get('point_sources')
    if not point_sources:
        raise ValueError(
            "No companion configured in default_param.yml's "
            "models.corgihowfsc.corgi_overrides.point_sources -- need a vmag "
            "(and a starting y position) to sweep."
        )
    vmag = point_sources[0]['vmag']
    y_mas = point_sources[0]['position_y_mas']

    host_overrides = without_point_sources(base_overrides)
    mgr_host = make_manager(cfg, hconf, mode, host_overrides)
    host_img = mgr_host.generate_host_star_psf(dm1v, dm2v, lind=lind, exptime=exptime)

    norm, peakflux = build_normalizer(cfg, hconf, mode, host_overrides, dm1v, dm2v, lind, exptime)

    dh_mask = insertinto(cfg.sl_list[lind].dh.e, host_img.shape).astype(bool)
    host_contrast = np.nanmean(norm.normalize(host_img, peakflux, exptime)[dh_mask])

    mas_per_lamd = calculate_mas_per_lamD(cfg.sl_list[lind].lam)

    x_positions = list(np.arange(x_start, x_stop, x_step))
    if not np.isclose(x_positions[-1], x_stop):
        x_positions.append(x_stop)

    frame_paths = []
    x_mas, c_combined, c_companion = [], [], []
    vmin = vmax = None

    frames_dir = os.path.join(out_dir, "dark_hole_frames")
    os.makedirs(frames_dir, exist_ok=True)

    for i, x in enumerate(x_positions):
        overrides = dict(base_overrides)
        overrides['point_sources'] = [{
            'vmag': vmag, 'position_x_mas': float(x), 'position_y_mas': float(y_mas),
        }]
        mgr_full = make_manager(cfg, hconf, mode, overrides)
        combined_img = mgr_full.generate_host_star_psf(dm1v, dm2v, lind=lind, exptime=exptime)
        companion_img = combined_img - host_img

        combined_ni = norm.normalize(combined_img, peakflux, exptime)
        companion_ni = norm.normalize(companion_img, peakflux, exptime)

        contrast_combined  = np.nanmean(combined_ni[dh_mask])
        contrast_companion = np.nanmean(companion_ni[dh_mask])

        x_mas.append(x)
        c_combined.append(contrast_combined)
        c_companion.append(contrast_companion)

        if vmin is None:
            # Fix the color scale from the first (closest-in, brightest) frame
            # so the GIF reads as motion rather than each frame rescaling itself.
            frame_clipped = clip_positive(combined_ni)
            vmin, vmax = np.nanmin(frame_clipped), np.nanmax(frame_clipped)

        lamd = x / mas_per_lamd
        save_frame_name = f'sweep_frame_{i:03d}_x{int(round(x))}mas'
        frame_path = os.path.join(frames_dir, save_frame_name + '.png')
        save_log_frame(combined_ni, frame_path, out_dir, save_frame_name, vmin, vmax,
                        title=f'x = {x:.0f} mas ({lamd:.1f} lam/D)')
        frame_paths.append(frame_path)
        print(f'[{i+1}/{len(x_positions)}] x={x:.0f} mas: '
              f'contrast(combined)={contrast_combined:.3e}, '
              f'contrast(companion only)={contrast_companion:.3e}')

    make_gif(frame_paths, os.path.join(out_dir, 'companion_sweep.gif'))
    iwa_lamd, owa_lamd = fov_range_lamd(mgr_host.cor_mapped)
    plot_contrast(np.array(x_mas), np.array(c_combined), np.array(c_companion),
                  host_contrast, mas_per_lamd, iwa_lamd, owa_lamd, mgr_host.cor_mapped,
                  os.path.join(out_dir, 'contrast_vs_separation.png'))
    # Save contrast info for plot
    np.savez("contrast_vs_xmas.npz", x_mas=x_mas, mas_per_lamd=mas_per_lamd, c_combined=np.array(c_combined), 
             c_companion=np.array(c_companion), host_contrast=host_contrast, iwa_lambd=iwa_lamd, owa_lamd=owa_lamd)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param_file', default=DEFAULT_PARAM_FILE,
                    help='Path to default_param.yml (default: corgihowfsc/scripts/default_param.yml)')
    ap.add_argument('--out-dir', default=DEFAULT_OUT_DIR,
                    help='Where to save figures/frames/gif')
    ap.add_argument('--sweep', action='store_true',
                    help='Sweep the companion x position instead of using a single frame')
    ap.add_argument('--x-start', type=float, default=415.0, help='Sweep start, mas')
    ap.add_argument('--x-stop', type=float, default=10000.0, help='Sweep stop, mas')
    ap.add_argument('--x-step', type=float, default=500.0, help='Sweep step, mas')
    ap.add_argument('--lind', type=int, default=0, help='Wavelength/subband index')
    ap.add_argument('--exptime', type=float, default=1.0,
                    help='Exposure time [s] used for image generation and NI normalization '
                         '(irrelevant while is_noise_free=True, since normalize() forces '
                         'exptime=1 in that case; matters once the noisy path is fixed)')
    ap.add_argument('--force-noise-free', action='store_true',
                    help="Override is_noise_free=True for this run only (doesn't touch "
                         "default_param.yml). Useful while the noisy detector path is "
                         "broken")
    args = ap.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_path = os.path.join(args.out_dir, f"{timestamp}_offAxisSweep")
    os.makedirs(folder_path, exist_ok=True)

    cfg, hconf, mode, dm1v, dm2v, base_overrides = build_inputs(args.param_file)
    if args.force_noise_free:
        base_overrides['is_noise_free'] = True

    if args.sweep:
        run_sweep(cfg, hconf, mode, dm1v, dm2v, base_overrides, args.lind, args.exptime,
                  folder_path, args.x_start, args.x_stop, args.x_step)
    else:
        run_single(cfg, hconf, mode, dm1v, dm2v, base_overrides, args.lind, args.exptime,
                  folder_path)


if __name__ == '__main__':
    main()
