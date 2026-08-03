"""
Stand-alone test for point-source companion injection in the CorgiSim GITL frames.

Purpose
-------
Verify that GitlImage.get_image() (corgihowfsc backend) generates an image with the
expected off-axis point-source companion, WITHOUT running any Jacobian / nulling loop.
It builds the same cfg / cstrat / hconf objects that the real run script uses, then
calls get_image() twice -- once with no companion and once with the companion in
corgi_overrides['point_sources'] -- and compares the two frames.

Run
---
    python sandbox/zja/test_point_source_companion.py

Edit the CONFIG block below to change the companion (vmag / position) or the mode.
"""

import os

# Headless-safe plotting: render to file, and also show a window if a display exists.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

# CorgiSim needs the Roman preflight PROPER prescription copied into the cwd.
import roman_preflight_proper
roman_preflight_proper.copy_here()

import corgihowfsc
from howfsc.control.cs import ControlStrategy
from howfsc.model.mode import CoronagraphMode
from howfsc.util.loadyaml import loadyaml

from corgihowfsc.utils.howfsc_initialization import get_args, load_files
from corgihowfsc.utils.corgisim_gitl_frames import GitlImage
from corgihowfsc.utils.corgisim_utils import calculate_mas_per_lamD


# --------------------------------------------------------------------------- #
# CONFIG -- edit these to taste
# --------------------------------------------------------------------------- #
MODE = 'wfov_band4'
DARK_HOLE = '360deg'

# The companion to inject. Matches the schema _build_point_source_info expects.
POINT_SOURCES = [
    {'vmag': 2.25, 'position_x_mas': 700.0, 'position_y_mas': 0.0},
]

# Keep this fast + clean: noise-free means get_image returns
# host_star_image + point_source_image directly (no detector realization).
IS_NOISE_FREE = True

EXPTIME = 1.0          # seconds; any positive value is fine when noise-free
GAIN = 1
LIND = 0               # wavelength/subband index (band 4 supports 0..2)
OUTPUT_DIM = 153       # detector cutout size (px), matches the GITL frame

OUTPNG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'point_source_companion_test.png')
# --------------------------------------------------------------------------- #


def build_inputs():
    """Load cfg / cstrat / hconf and the starting DM maps, exactly as the run_corgisim_nulling_gitl
    script does, but without any Jacobian /etxtra stuff that slows the loop down."""
    howfscpath = os.path.dirname(os.path.abspath(corgihowfsc.__file__))

    args = get_args(
        mode=MODE,
        dark_hole=DARK_HOLE,
        probe_shape='default',
        jacpath=None,          # no Jacobian needed for image generation
    )

    (modelpath, cfgfile, jacfile, cstratfile, probefiles,
     hconffile, n2clistfiles, dmstartmaps) = load_files(args, howfscpath)

    cfg = CoronagraphMode(cfgfile)
    cstrat = ControlStrategy(cstratfile)
    hconf = loadyaml(hconffile, custom_exception=TypeError)

    dm1v, dm2v = dmstartmaps[0], dmstartmaps[1]
    return cfg, cstrat, hconf, dm1v, dm2v


def make_image(cfg, cstrat, hconf, dm1v, dm2v, point_sources):
    """Build a corgihowfsc GitlImage and return one get_image() frame."""
    corgi_overrides = {
        'output_dim': OUTPUT_DIM,
        'is_noise_free': IS_NOISE_FREE,
    }
    if point_sources:
        corgi_overrides['point_sources'] = point_sources

    imager = GitlImage(
        cfg=cfg,
        cstrat=cstrat,
        hconf=hconf,
        backend='corgihowfsc',
        cor=MODE,
        corgi_overrides=corgi_overrides,
    )

    return imager.get_image(
        dm1v=dm1v,
        dm2v=dm2v,
        exptime=EXPTIME,
        gain=GAIN,
        lind=LIND,
    )


def main():
    cfg, cstrat, hconf, dm1v, dm2v = build_inputs()

    # Report the expected companion geometry so the plot is interpretable.
    lam = cfg.sl_list[LIND].lam
    mas_per_lamD = calculate_mas_per_lamD(lam)
    print(f"Mode {MODE} / lind {LIND}: lam = {lam*1e9:.1f} nm, "
          f"{mas_per_lamD:.2f} mas per lambda/D")
    for src in POINT_SOURCES:
        r_mas = np.hypot(src['position_x_mas'], src['position_y_mas'])
        print(f"  companion vmag={src['vmag']} at "
              f"(dx={src['position_x_mas']}, dy={src['position_y_mas']}) mas "
              f"= {r_mas / mas_per_lamD:.2f} lambda/D from host")

    print("Generating host-only frame...")
    img_host = make_image(cfg, cstrat, hconf, dm1v, dm2v, point_sources=None)

    print("Generating frame WITH companion...")
    img_comp = make_image(cfg, cstrat, hconf, dm1v, dm2v, point_sources=POINT_SOURCES)

    diff = img_comp - img_host

    # --- Verification: the companion should add localized, off-axis flux ------
    assert img_host.shape == img_comp.shape, "frame shapes differ"
    center = np.array(diff.shape) / 2.0

    added_flux = np.nansum(diff)
    peak_yx = np.unravel_index(np.nanargmax(diff), diff.shape)
    peak_offset_px = np.hypot(peak_yx[0] - center[0], peak_yx[1] - center[1])

    print(f"\nTotal flux added by companion: {added_flux:.4g}")
    print(f"Brightest new-flux pixel at (row, col) = {peak_yx}, "
          f"{peak_offset_px:.1f} px from center")

    assert added_flux > 0, "companion added no flux -- point source not injected"
    assert peak_offset_px > 3.0, (
        "brightest added flux is at the center -- companion not off-axis as expected"
    )
    print("PASS: companion appears as localized off-axis flux.\n")

    # --- Plot for visual confirmation ----------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, data, title in zip(
        axes,
        [img_host, img_comp, diff],
        ['Host only', 'Host + companion', 'Difference (companion)'],
    ):
        finite = data[np.isfinite(data)]
        vmax = np.nanpercentile(finite, 99.5) if finite.size else 1.0
        im = ax.imshow(data, origin='lower', vmax=vmax)
        ax.set_title(title)
        ax.plot(center[1], center[0], '+', color='red', ms=10)  # host location
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"{MODE} point-source companion test (noise_free={IS_NOISE_FREE})")
    fig.tight_layout()
    fig.savefig(OUTPNG, dpi=120)
    print(f"Saved figure to {OUTPNG}")
    try:
        plt.show()
    except Exception:
        pass


if __name__ == '__main__':
    main()
