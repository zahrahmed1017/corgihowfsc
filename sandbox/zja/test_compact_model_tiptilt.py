import numpy as np
import matplotlib.pyplot as plt
from synphot import Observation
from corgisim import scene
from howfsc.model.mode import CoronagraphMode
from howfsc.util.loadyaml import loadyaml
from howfsc.util.prop_tools import open_efield
from corgihowfsc.utils.corgisim_manager import CorgisimManager
from corgihowfsc.utils.cgi_prop_tools import (
    mas_to_tiptilt, build_offaxis_cfg, build_broadband_offaxis_cfg
)

CFGFILE = 'corgihowfsc/model/wfov_mswc_band4a/wfov_mswc_band4a_5x5lamD/howfsc_optical_model.yaml'
HCONFFILE = 'corgihowfsc/model/wfov_mswc_band4a/hconf_wfov_mswc_band4a.yaml'
IND = 0
DX_MAS, DY_MAS = 7000.0, 0

def peak_offset(img):
    cr, cc = img.shape[0] // 2, img.shape[1] // 2
    r, c = np.unravel_index(np.argmax(img), img.shape)
    return r - cr, c - cc

def broadband_compact_model(cfg, ind, dm1v, dm2v, dx_mas, dy_mas, optics, spectrum):
    """
    Sum monochromatic compact-model (open_efield) intensities over the same
    wavelength samples corgisim used for its subband ('optics.lam_um'),
    weighted by the same off-axis-source photon counts per wavelength bin
    that corgisim uses internally (instrument.CorgiOptics.inject_point_sources),
    so the two models are compared on an apples-to-apples broadband basis.

    Reuses channel `ind`'s masks unchanged at each sampled wavelength
    (achromatic-mask-within-a-subband approximation); see
    build_broadband_offaxis_cfg for what that does and does not capture.
    """
    lam_um = optics.lam_um
    dlam_um = lam_um[1] - lam_um[0]
    obs = Observation(spectrum, optics.bp)

    weights = np.array([
        obs.countrate(
            area=optics.area,
            waverange=[(lam - 0.5 * dlam_um) * 1e4, (lam + 0.5 * dlam_um) * 1e4],
        ).value
        for lam in lam_um
    ])

    I_per_lam = []
    for lam in lam_um:
        offaxis_cfg = build_broadband_offaxis_cfg(cfg, ind, lam * 1e-6, dx_mas, dy_mas)
        I_per_lam.append(np.abs(open_efield(offaxis_cfg, [dm1v, dm2v], ind)) ** 2)
    I_broadband = np.average(np.array(I_per_lam), axis=0, weights=weights)
    return I_broadband, lam_um, weights

def main():
    cfg = CoronagraphMode(CFGFILE)
    hconf = loadyaml(HCONFFILE, custom_exception=TypeError)
    dm1v = np.zeros((cfg.dmlist[0].registration['nact'],) * 2)
    dm2v = np.zeros((cfg.dmlist[1].registration['nact'],) * 2)

    sl = cfg.sl_list[IND]
    tip_pred, tilt_pred = mas_to_tiptilt(DX_MAS, DY_MAS, sl)
    print(f"predicted tip(col)={tip_pred:.2f} tilt(row)={tilt_pred:.2f}")

    offaxis_cfg = build_offaxis_cfg(cfg, DX_MAS, DY_MAS)
    I_compact_mono = np.abs(open_efield(offaxis_cfg, [dm1v, dm2v], IND)) ** 2
    print("compact model (monochromatic) peak offset (row,col):", peak_offset(I_compact_mono))

    manager = CorgisimManager(
        cfg=cfg, cstrat=None, hconf=hconf, cor='wfov_mswc_band4a',
        corgi_overrides={
            'bandpass': '4', 'host_star_enabled': False,
            'point_sources': [{'vmag': 5.0, 'position_x_mas': DX_MAS, 'position_y_mas': DY_MAS}],
            'is_noise_free': True, 'output_dim': I_compact_mono.shape[0],
            'use_fpm': 0, 'NCPUS': 1,
        },
    )
    I_corgisim = manager.generate_host_star_psf(dm1v, dm2v, lind=IND)
    print("corgisim (broadband subband) peak offset (row,col):", peak_offset(I_corgisim))

    # Build the compact-model broadband image over the same wavelength grid
    # and per-wavelength weights corgisim used for its off-axis point source.
    optics = manager.create_optics(dm1v, dm2v, IND)
    offaxis_scene = scene.Scene(manager.host_star_properties, manager.point_sources)
    spectrum = offaxis_scene.off_axis_source_spectrum[0]
    I_compact_broadband, lam_um, weights = broadband_compact_model(
        cfg, IND, dm1v, dm2v, DX_MAS, DY_MAS, optics, spectrum
    )
    print(f"compact model broadband sampled at {len(lam_um)} wavelengths "
          f"[{lam_um.min():.4f}, {lam_um.max():.4f}] um, "
          f"relative weights: {np.round(weights / weights.sum(), 3)}")
    print("compact model (broadband) peak offset (row,col):", peak_offset(I_compact_broadband))

    imgs = [I_compact_broadband, I_corgisim]
    titles = ['compact model, broadband sum (open_efield)', 'corgisim/PROPER (broadband)']
    log_imgs = [np.log10(img / img.max() + 1e-10) for img in imgs]
    vmax = max(im.max() for im in log_imgs)
    vmin = max(im.min() for im in log_imgs)  # tightest floor common to both, so noise floor isn't washed out

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, log_img, img, title in zip(axes, log_imgs, imgs, titles):
        im = ax.imshow(log_img, origin='lower', vmin=vmin, vmax=vmax)
        cr, cc = img.shape[0] // 2, img.shape[1] // 2
        ax.plot(cc, cr, 'w+', ms=12, label='on-axis center')
        ax.plot(cc + tip_pred, cr + tilt_pred, 'rx', ms=12, label='predicted (tip,tilt)')
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='log10(Normalized Intensity)')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()