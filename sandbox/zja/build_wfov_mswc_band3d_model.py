"""
Build the wfov_mswc_band3d compact-model folder from wfov_mswc_band4a.

Same physical masks as band4a (MSWC SPM, WFOV/MSWC FPM, Lyot), only the color
filter changes to band 3d (cgisim: 750-758 nm). What changes:
  - per-subband pupil phase files: they are achromatic OPD, so phase scales as lam4a/lam3d
  - fpm.ppl and fs.ppl scale as lam3d/lam4a (fixed physical mask, pixels per lam/D grows with lam)
  - DH mask regenerated at the new image-plane ppl, same geometry in lam/D
    (5x5 lam/D square centered at (10, -10) lam/D)

DH mask builder is a port of build_rounded_square_mask() in generate_temporary_masks.m.

Run from the corgihowfsc repo root:
    python sandbox/zja/build_wfov_mswc_band3d_model.py
"""
import os
import shutil

import numpy as np
import yaml
from astropy.io import fits

MODEL_DIR = 'corgihowfsc/model'
SRC_BAND = os.path.join(MODEL_DIR, 'wfov_mswc_band4a')
DST_BAND = os.path.join(MODEL_DIR, 'wfov_mswc_band3d')
SRC_DH = os.path.join(SRC_BAND, 'wfov_mswc_band4a_5x5lamD')
DST_DH = os.path.join(DST_BAND, 'wfov_mswc_band3d_5x5lamD')

LAM_4A = 7.914113988892523e-07
LAM_3D = 7.54e-07  # center of cgisim band 3d (0.750-0.758 um)
SCALE = LAM_3D / LAM_4A

NPIX = 153
DH_HALFWIDTH_LAMD = 2.5
DH_XOFFSET_LAMD = 10.0
DH_YOFFSET_LAMD = -10.0


def build_square_mask(npix, ppl, hw_x_lamd, hw_y_lamd, x_off_lamd, y_off_lamd):
    """Hard-edged square mask, same conventions as the MATLAB version (no fillet, no rotation)."""
    c = (npix + 1) / 2
    x, y = np.meshgrid(np.arange(1, npix + 1) - c, np.arange(1, npix + 1) - c)
    qx = np.abs(x - x_off_lamd * ppl) - hw_x_lamd * ppl
    qy = np.abs(y - y_off_lamd * ppl) - hw_y_lamd * ppl
    d = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0)
    return (d <= 0).astype(np.float32)


def write_dh_yaml(filename, ppl):
    with open(filename, 'w') as f:
        f.write(f'nRows: {NPIX}\n')
        f.write(f'nCols: {NPIX}\n')
        f.write('shapes:\n')
        f.write('  shape0:\n')
        f.write('    shape: square\n')
        f.write(f'    halfWidthXLamD: {DH_HALFWIDTH_LAMD:.6f}\n')
        f.write(f'    halfWidthYLamD: {DH_HALFWIDTH_LAMD:.6f}\n')
        f.write(f'    xOffsetLamD: {DH_XOFFSET_LAMD:.6f}\n')
        f.write(f'    yOffsetLamD: {DH_YOFFSET_LAMD:.6f}\n')
        f.write(f'    ppl: {ppl:.10f}\n')
        f.write(f'    halfWidthXPix: {DH_HALFWIDTH_LAMD * ppl:.6f}\n')
        f.write(f'    halfWidthYPix: {DH_HALFWIDTH_LAMD * ppl:.6f}\n')
        f.write(f'    xOffsetPix: {DH_XOFFSET_LAMD * ppl:.6f}\n')
        f.write(f'    yOffsetPix: {DH_YOFFSET_LAMD * ppl:.6f}\n')


def main():
    with open(os.path.join(SRC_DH, 'howfsc_optical_model.yaml')) as f:
        cfg = yaml.safe_load(f)
    sl = cfg['sls'][0]
    ppl_fs_4a = sl['fs']['ppl']

    # Sanity check: the ported mask builder reproduces the band4a DH mask exactly
    dh_4a = fits.getdata(os.path.join(SRC_DH, 'subband0', 'dh_sw_mask.fits'))
    dh_4a_check = build_square_mask(NPIX, ppl_fs_4a, DH_HALFWIDTH_LAMD, DH_HALFWIDTH_LAMD,
                                    DH_XOFFSET_LAMD, DH_YOFFSET_LAMD)
    if not np.array_equal(dh_4a, dh_4a_check):
        raise RuntimeError('Ported DH mask builder does not reproduce band4a dh_sw_mask.fits')

    # any/: amplitude files unchanged, phase files rescaled.
    # fs_amp.fits is skipped: it was built at the band4a ppl and band3d runs without a field stop.
    os.makedirs(os.path.join(DST_BAND, 'any', 'subband0'), exist_ok=True)
    for fn in os.listdir(os.path.join(SRC_BAND, 'any')):
        src = os.path.join(SRC_BAND, 'any', fn)
        if os.path.isfile(src) and fn != 'fs_amp.fits':
            shutil.copy2(src, os.path.join(DST_BAND, 'any', fn))
    for fn in ['epup_ph.fits', 'lyot_ph.fits', 'spm_ph.fits']:
        ph = fits.getdata(os.path.join(SRC_BAND, 'any', 'subband0', fn))
        fits.writeto(os.path.join(DST_BAND, 'any', 'subband0', fn), ph * (LAM_4A / LAM_3D),
                     overwrite=True)

    # dark-hole folder
    os.makedirs(os.path.join(DST_DH, 'subband0'), exist_ok=True)
    for fn in ['gitl_start_compact_dm1.fits', 'gitl_start_compact_dm2.fits']:
        shutil.copy2(os.path.join(SRC_DH, fn), os.path.join(DST_DH, fn))
    shutil.copy2(os.path.join(SRC_DH, 'cstrat_wfov_mswc_band4a.yaml'),
                 os.path.join(DST_DH, 'cstrat_wfov_mswc_band3d.yaml'))

    # optical model yaml: edit only lam and the two ppl values, textually, to keep formatting
    with open(os.path.join(SRC_DH, 'howfsc_optical_model.yaml')) as f:
        text = f.read()
    ppl_fpm_4a = sl['fpm']['ppl']
    ppl_fs_3d = ppl_fs_4a * SCALE
    ppl_fpm_3d = ppl_fpm_4a * SCALE
    for old, new in [(f'lam: {LAM_4A!r}', f'lam: {LAM_3D!r}'),
                     (f'ppl: {ppl_fs_4a!r}', f'ppl: {ppl_fs_3d!r}'),
                     (f'ppl: {ppl_fpm_4a!r}', f'ppl: {ppl_fpm_3d!r}')]:
        if text.count(old) != 1:
            raise RuntimeError(f'Expected exactly one "{old}" in band4a optical model yaml')
        text = text.replace(old, new)
    with open(os.path.join(DST_DH, 'howfsc_optical_model.yaml'), 'w') as f:
        f.write(text)

    # DH mask at the new image-plane ppl
    dh_3d = build_square_mask(NPIX, ppl_fs_3d, DH_HALFWIDTH_LAMD, DH_HALFWIDTH_LAMD,
                              DH_XOFFSET_LAMD, DH_YOFFSET_LAMD)
    fits.writeto(os.path.join(DST_DH, 'subband0', 'dh_sw_mask.fits'), dh_3d, overwrite=True)
    write_dh_yaml(os.path.join(DST_DH, 'subband0', 'dh_sw_mask.yaml'), ppl_fs_3d)

    # hconf
    with open(os.path.join(SRC_BAND, 'hconf_wfov_mswc_band4a.yaml')) as f:
        text = f.read()
    text = text.replace('sequence_list: [WFOV_WFOV_4A_SEED]', 'sequence_list: [WFOV_WFOV_3D_SEED]')
    text = text.replace('sequence_observation: WFOV_WFOV_4_SEED', 'sequence_observation: WFOV_WFOV_3D_SEED')
    with open(os.path.join(DST_BAND, 'hconf_wfov_mswc_band3d.yaml'), 'w') as f:
        f.write(text)

    print(f'lam: {LAM_3D}, fs.ppl: {ppl_fs_3d}, fpm.ppl: {ppl_fpm_3d}')
    print(f'DH pixels: band4a {int(dh_4a.sum())}, band3d {int(dh_3d.sum())}')


if __name__ == '__main__':
    main()
