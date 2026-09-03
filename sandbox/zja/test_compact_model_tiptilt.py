import numpy as np
import matplotlib.pyplot as plt
from howfsc.model.mode import CoronagraphMode
from howfsc.util.loadyaml import loadyaml
from howfsc.util.prop_tools import open_efield
from corgihowfsc.utils.corgisim_manager import CorgisimManager
from corgihowfsc.utils.cgi_prop_tools import mas_to_tiptilt, build_offaxis_cfg

CFGFILE = 'corgihowfsc/model/wfov_mswc_band4a/wfov_mswc_band4a_5x5lamD/howfsc_optical_model.yaml'
HCONFFILE = 'corgihowfsc/model/wfov_mswc_band4a/hconf_wfov_mswc_band4a.yaml'
IND = 0
DX_MAS, DY_MAS = 7000.0, 0 

def peak_offset(img):
    cr, cc = img.shape[0] // 2, img.shape[1] // 2
    r, c = np.unravel_index(np.argmax(img), img.shape)
    return r - cr, c - cc

def main():
    cfg = CoronagraphMode(CFGFILE)
    hconf = loadyaml(HCONFFILE, custom_exception=TypeError)
    dm1v = np.zeros((cfg.dmlist[0].registration['nact'],) * 2)
    dm2v = np.zeros((cfg.dmlist[1].registration['nact'],) * 2)

    sl = cfg.sl_list[IND]
    tip_pred, tilt_pred = mas_to_tiptilt(DX_MAS, DY_MAS, sl)
    print(f"predicted tip(col)={tip_pred:.2f} tilt(row)={tilt_pred:.2f}")

    offaxis_cfg = build_offaxis_cfg(cfg, DX_MAS, DY_MAS)
    I_compact = np.abs(open_efield(offaxis_cfg, [dm1v, dm2v], IND)) ** 2
    print("compact model peak offset (row,col):", peak_offset(I_compact))

    manager = CorgisimManager(
        cfg=cfg, cstrat=None, hconf=hconf, cor='wfov_mswc_band4a',
        corgi_overrides={
            'bandpass': '4', 'host_star_enabled': False,
            'point_sources': [{'vmag': 5.0, 'position_x_mas': DX_MAS, 'position_y_mas': DY_MAS}],
            'is_noise_free': True, 'output_dim': I_compact.shape[0],
            'use_fpm': 0, 'NCPUS': 1,
        },
    )
    I_corgisim = manager.generate_host_star_psf(dm1v, dm2v, lind=IND)
    print("corgisim peak offset (row,col):", peak_offset(I_corgisim))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, img, title in zip(axes, [I_compact, I_corgisim], ['compact model (open_efield)', 'corgisim/PROPER']):
        ax.imshow(np.log10(img / img.max() + 1e-10), origin='lower')
        cr, cc = img.shape[0] // 2, img.shape[1] // 2
        ax.plot(cc, cr, 'w+', ms=12, label='on-axis center')
        ax.plot(cc + tip_pred, cr + tilt_pred, 'rx', ms=12, label='predicted (tip,tilt)')
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=7)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()