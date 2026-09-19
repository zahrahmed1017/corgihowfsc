"""
Diagnostic: does the off-axis (SNWC/MSWC) compact-model Jacobian G correctly
predict how the dark-hole field responds to a DM move, and does inverting G
actually null a known, deliberately-injected aberration?

Script: run corgihowfsc.gitl.nulling_gitl() twice (once from a flat DM baseline, once from baseline + a known sine-ripple DM
perturbation `a`) everything else stays the same, and reads the results back from the
same per-iteration FITS files corgihowfsc/utils/saving_output.py always
writes. That way normalization, exposure-time calculation, estimation,
and the control-law inversion are all done exactly how they are in the full run_corgisim_nulling_gitl.py   

The only piece not included in any output file is the actual Jacobian array
itself so tthis script also does one extra
call to howfsc.precomp.howfsc_precomputation with the same arguments
nulling_gitl() uses internally, and then saves the G matrix for analysis.

"""

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits

from howfsc.control.cs import ControlStrategy
from howfsc.model.mode import CoronagraphMode
from howfsc.util.loadyaml import loadyaml
from howfsc.util.insertinto import insertinto
from howfsc.util.prop_tools import efield as compact_efield
from howfsc.precomp import howfsc_precomputation

import corgihowfsc
from corgihowfsc.utils.howfsc_initialization import get_args, load_files
from corgihowfsc.sensing.PerfectEstimator import PerfectEstimator
from corgihowfsc.sensing.GettingProbes import ProbesShapes
from corgihowfsc.utils.contrast_nomalization import EETCNormalization
from corgihowfsc.gitl.nulling_gitl import nulling_gitl
from corgihowfsc.utils.corgisim_gitl_frames import GitlImage
from corgihowfsc.utils.corgisim_utils import resolve_field_stop_array
from corgihowfsc.utils.cgi_prop_tools import build_offaxis_cfg

HOWFSCPATH = os.path.dirname(os.path.abspath(corgihowfsc.__file__))
PARAM_FILE = os.path.join(HOWFSCPATH, 'scripts', 'default_param.yml')

SCRATCH_ROOT = os.environ.get(
    'OFFAXIS_TEST_SCRATCH',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '_offaxis_jac_test_output'),
)

# Sine-ripple test actuation `a`: amplitude in [0.1, 1] rad (large enough to
# see clearly, small enough for the linear model to be valid), spatial
# frequency so the resulting speckle lands inside the dark hole (10 lam/D away from on-axis) 
RIPPLE_AMP_RAD = 0.2
RIPPLE_FX = 20
RIPPLE_FY = -10

TOL_STEP2_RELERR = 0.30
TOL_STEP4_REDUCTION = 5.0

params = loadyaml(PARAM_FILE)
ACTIVE_MODEL = params['active_model']
SIM_SETTINGS = params['sim_settings']
CROP_CFG = params['crop']
MODEL_CFG = params['models'][ACTIVE_MODEL]

MODE = SIM_SETTINGS['mode']
DARK_HOLE = SIM_SETTINGS['dark_hole']
PROBE_SHAPE = SIM_SETTINGS['probe_shape']
PRECOMP = SIM_SETTINGS['precomp']
BACKEND_TYPE = MODEL_CFG['backend_type']
DEFAULT_DMSTARTMAP_FILENAMES = list(MODEL_CFG['dmstartmap_filenames'])
CORGI_OVERRIDES_BASE = MODEL_CFG.get('corgi_overrides', {}).copy()
STARTING_CONTRAST = float(MODEL_CFG['starting_contrast'])
DX_MAS = CORGI_OVERRIDES_BASE['point_sources'][0]['position_x_mas']
DY_MAS = CORGI_OVERRIDES_BASE['point_sources'][0]['position_y_mas']


def build_run(niter, dmstartmap_filenames, run_dir):
    """Build all objects the same way run_corgisim_nulling_gitl.py::main()
    does, then call the real nulling_gitl() production loop. Returns the
    objects needed to read the run back and to redo the Jacobian side-call."""
    args = get_args(
        niter=niter,
        mode=MODE,
        dark_hole=DARK_HOLE,
        probe_shape=PROBE_SHAPE,
        precomp=PRECOMP,
        num_process=1,
        num_threads=1,
        fileout=os.path.join(run_dir, 'final_frames.fits'),
        jacpath=None,
        dmstartmap_filenames=dmstartmap_filenames,
        logfile=os.path.join(run_dir, 'gitl.log'),
    )
    args.starting_contrast = STARTING_CONTRAST
    args.num_imager_worker = 1
    args.num_proper_process = 1
    args.use_mpi = False
    args.debug = False
    args.mpi_comm = None

    modelpath, cfgfile, jacfile, cstratfile, probefiles, hconffile, \
        n2clistfiles, dmstartmaps = load_files(args, HOWFSCPATH)

    cfg = CoronagraphMode(cfgfile)
    cfg = build_offaxis_cfg(cfg, DX_MAS, DY_MAS)

    hconf = loadyaml(hconffile, custom_exception=TypeError)
    cstrat = ControlStrategy(cstratfile)
    probes = ProbesShapes(args.probe_shape)

    crop_params = {
        'nrow': CROP_CFG['nrow'],
        'ncol': CROP_CFG['ncol'],
        'lrow': MODEL_CFG['lrow'],
        'lcol': MODEL_CFG['lcol'],
    }

    corgi_overrides = CORGI_OVERRIDES_BASE.copy()
    corgi_overrides['output_dim'] = crop_params['nrow']
    corgi_overrides = resolve_field_stop_array(corgi_overrides, cfg, modelpath)

    imager = GitlImage(
        cfg=cfg, cstrat=cstrat, hconf=hconf,
        backend=BACKEND_TYPE, cor=MODE, corgi_overrides=corgi_overrides,
    )

    # Perfect estimator: meas_efield is then the direct corgisim/PROPER
    # field, not a pairwise-probing reconstruction -- keeps "is G right"
    # decoupled from "does probe-based estimation work". Also reduce probe
    # pairs to 1, matching run_corgisim_nulling_gitl.py's own speedup.
    estimator = PerfectEstimator()
    probefiles = {0: probefiles[0]}
    hconf['probe']['dmrel_ph_list'] = hconf['probe']['dmrel_ph_list'][:1]

    normalization_strategy = EETCNormalization(BACKEND_TYPE, corgi_overrides)

    metadata = {'diagnostic': 'test_offaxis_jacobian_control', 'niter': niter}

    nulling_gitl(
        cstrat, estimator, probes, normalization_strategy, imager,
        cfg, args, hconf, modelpath, jacfile, probefiles, n2clistfiles,
        crop_params, dmstartmaps, metadata=metadata, output_every_iter=True,
    )

    return {
        'cfg': cfg, 'cstrat': cstrat, 'crop_params': crop_params,
        'dmstartmaps': dmstartmaps, 'modelpath': modelpath, 'run_dir': run_dir,
    }


def read_meas_efield(run_dir, iteration_idx):
    """Read back other[0]['meas_efield'] for a given iteration, as written
    by save_outputs_iter() -> efield_estimations.fits (real/imag interleaved
    planes, one pair per wavelength; nlam=1 here so shape is (2, nrow, ncol))."""
    path = os.path.join(run_dir, f'iteration_{iteration_idx:04d}', 'efield_estimations.fits')
    data = pyfits.getdata(path, ext=1)
    return data[0] + 1j * data[1]


def read_dm_commands(run_dir, iteration_idx):
    dm1 = pyfits.getdata(os.path.join(run_dir, f'iteration_{iteration_idx:04d}', 'dm1_command.fits'))
    dm2 = pyfits.getdata(os.path.join(run_dir, f'iteration_{iteration_idx:04d}', 'dm2_command.fits'))
    return dm1, dm2


def finite_diff_selfcheck(cfg, dm10, dm20, G, nact, lamc, dh_mask,
                           n_test_actuators=5, delta=0.01, corr_tol=0.9):
    """
    Fast (seconds, compact-model-only, no corgisim) sanity gate. For each of
    the n_test_actuators DM1 actuators with the largest Jacobian column norm,
    runs TWO different comparisons against a direct single-actuator
    finite-difference derivative of the compact model's own nonlinear
    propagation (howfsc.util.prop_tools.efield):

    1. "full G" vs the finite-difference. NOT strictly apples-to-apples:
       per howfsc.control.calcjacs.calcjacs_sp, G's column is actually
       edhv/epk0 - edhv0/epk0**2*epk, a "joint DM + peak/Strehl" Jacobian,
       not a plain field-sensitivity -- while the finite-difference here is
       of the raw (unnormalized) field only. A mismatch here is still
       meaningful (jac_solve/inv_to_dm use this exact G for the control
       step), but doesn't by itself localize the fault to the sensitivity
       computation vs. the peak-normalization term.
    2. "raw edhv" vs the same finite-difference. Apples-to-apples: edhv is
       pokeprop()'s direct output (via proptolyot/proptodh), taken *before*
       calcjacs_sp ever applies epk0/epk. This isolates whether the core
       differential-propagation machinery (pokeprop, get_jac_precomp) is
       itself correct, fully decoupled from the peak/Strehl-normalization
       term. If (1) fails but (2) passes, the fault is specifically in
       epk0/epk (SingleLambda.get_inorm's mean()-based peak extraction,
       which only works for an on-axis/untitled field -- see
       OFFAXIS_JACOBIAN_PLAN.md), not in the sensitivity computation.

    Returns (all_pass_full, all_pass_raw, results) where results is a list
    of (idx, row, col, jac_col_norm, fd_norm, ratio_full, corr_full,
    edhv_norm, ratio_raw, corr_raw).
    """
    col_norms = np.linalg.norm(G[:nact * nact, :], axis=1)
    test_idxs = np.argsort(col_norms)[::-1][:n_test_actuators]

    dm10_height = cfg.dmlist[0].dmvobj.volts_to_dmh(dm10, lamc)
    E0 = compact_efield(cfg, [dm10, dm20], 0)

    sl = cfg.sl_list[0]
    sl.get_jac_precomp([dm10, dm20])  # required precompute for pokeprop()

    results = []
    all_pass_full = True
    all_pass_raw = True
    for idx in test_idxs:
        r, c = int(idx) // nact, int(idx) % nact
        h_single = np.zeros((nact, nact))
        h_single[r, c] = delta
        dm1v_pert = cfg.dmlist[0].dmvobj.dmh_to_volts(dm10_height + h_single, lamc)
        E1 = compact_efield(cfg, [dm1v_pert, dm20], 0)
        dE_fd = (E1 - E0)[dh_mask] / delta
        fd_norm = np.linalg.norm(dE_fd)

        # Check 1: full G (what jac_solve/inv_to_dm actually use)
        jac_col = G[idx, :]
        jac_norm = np.linalg.norm(jac_col)
        ratio_full = fd_norm / jac_norm if jac_norm > 0 else np.inf
        corr_full = (np.vdot(jac_col, dE_fd) / (jac_norm * fd_norm)
                     if jac_norm > 0 and fd_norm > 0 else 0.0)
        passed_full = abs(corr_full) > corr_tol and 0.5 < ratio_full < 2.0
        all_pass_full = all_pass_full and passed_full

        # Check 2: raw edhv, apples-to-apples, pre-epk0/peak-term
        edm = sl.pokeprop(int(idx), [dm10, dm20])
        ely = sl.proptolyot(edm)
        edh = sl.proptodh(ely)
        edhv_raw = edh.ravel()[sl.dh_inds]
        edhv_norm = np.linalg.norm(edhv_raw)
        ratio_raw = fd_norm / edhv_norm if edhv_norm > 0 else np.inf
        corr_raw = (np.vdot(edhv_raw, dE_fd) / (edhv_norm * fd_norm)
                    if edhv_norm > 0 and fd_norm > 0 else 0.0)
        passed_raw = abs(corr_raw) > corr_tol and 0.5 < ratio_raw < 2.0
        all_pass_raw = all_pass_raw and passed_raw

        results.append((int(idx), r, c, jac_norm, fd_norm, ratio_full, corr_full,
                         edhv_norm, ratio_raw, corr_raw))
        print(f'  actuator idx={idx} (r={r},c={c}):')
        print(f'    [1: full G, includes epk0/peak-term] ||jac_col||={jac_norm:.4g}, '
              f'||finite-diff||={fd_norm:.4g}, ratio={ratio_full:.4g}, '
              f'correlation={corr_full:.4g} {"OK" if passed_full else "MISMATCH"}')
        print(f'    [2: raw edhv, apples-to-apples]       ||edhv_raw||={edhv_norm:.4g}, '
              f'||finite-diff||={fd_norm:.4g}, ratio={ratio_raw:.4g}, '
              f'correlation={corr_raw:.4g} {"OK" if passed_raw else "MISMATCH"}')

    return all_pass_full, all_pass_raw, results


def main():
    os.makedirs(SCRATCH_ROOT, exist_ok=True)

    # --- Step 0: resolve modelpath + baseline dmstartmaps and grab nact/lamc off a throwaway
    # off-axis cfg for building the sine-ripple perturbation below.
    probe_args = get_args(
        niter=1, mode=MODE, dark_hole=DARK_HOLE, probe_shape=PROBE_SHAPE,
        precomp=PRECOMP, dmstartmap_filenames=DEFAULT_DMSTARTMAP_FILENAMES,
        fileout=os.path.join(SCRATCH_ROOT, 'unused.fits'),
    )
    modelpath, cfgfile, _, cstratfile, _, _, _, dmstartmaps_baseline = load_files(probe_args, HOWFSCPATH)
    dm10, dm20 = dmstartmaps_baseline

    cfg0 = build_offaxis_cfg(CoronagraphMode(cfgfile), DX_MAS, DY_MAS)
    nact = cfg0.dmlist[0].registration['nact']
    dm2_nact = cfg0.dmlist[1].registration['nact']
    lamc = float(np.median([sl.lam for sl in cfg0.sl_list]))

    print(f'Off-axis star at ({DX_MAS}, {DY_MAS}) mas; DM1 nact={nact}, lamc={lamc:.4e} m')

    # --- TEST 1: fast compact-model self-consistency gate.
    # This will help us understand if G itself is a correct linearization of the compact
    # model's nonlinear propagation for this nonlinear config. 
    cstrat0 = ControlStrategy(cstratfile)
    crop_params = {
        'nrow': CROP_CFG['nrow'], 'ncol': CROP_CFG['ncol'],
        'lrow': MODEL_CFG['lrow'], 'lcol': MODEL_CFG['lcol'],
    }
    subcroplist = [(crop_params['lrow'], crop_params['lcol'], crop_params['nrow'], crop_params['ncol'])]

    print('Computing baseline Jacobian for the self-consistency gate...')
    jac0, jtwj_map0, _ = howfsc_precomputation(
        cfg=cfg0, dmset_list=[dm10, dm20], cstrat=cstrat0,
        subcroplist=subcroplist, jacmethod='fast', do_n2clist=False,
    )
    G0 = jac0[0] + 1j * jac0[1]
    dh_mask0 = insertinto(cfg0.sl_list[0].dh.e, (crop_params['nrow'], crop_params['ncol'])).astype(bool)

    print('[TEST 1] finite-difference self-consistency check '
          '(compact-model Jacobian vs compact-model nonlinear propagation) -- '
          'two comparisons per actuator, see finite_diff_selfcheck() docstring:')
    gate_pass_full, gate_pass_raw, gate_results = finite_diff_selfcheck(
        cfg0, dm10, dm20, G0, nact, lamc, dh_mask0,
    )
    print(f'[TEST 1] check 1 (full G, what jac_solve actually uses): '
          f'{"PASS" if gate_pass_full else "FAIL"}')
    print(f'[TEST 1] check 2 (raw edhv, apples-to-apples, isolates pokeprop): '
          f'{"PASS" if gate_pass_raw else "FAIL"}')
    # if not gate_pass_full:
    #     if gate_pass_raw:
    #         print(
    #             '[TEST 1] check 1 fails but check 2 passes -- this localizes '
    #             'the fault specifically to the epk0/epk peak-Strehl-correction '
    #             'term in howfsc.control.calcjacs.calcjacs_sp (in turn caused by '
    #             'SingleLambda.get_inorm()\'s mean()-based peak extraction, which '
    #             'only works for an on-axis/untitled field). The core '
    #             'differential-propagation machinery (pokeprop/get_jac_precomp) '
    #             'is confirmed correct for this off-axis config.'
    #         )
    #     else:
    #         print(
    #             '[TEST 1] both checks fail -- the fault is not isolated to '
    #             'epk0/epk; the core pokeprop/get_jac_precomp sensitivity '
    #             'computation itself disagrees with the nonlinear propagation.'
    #         )
    #     print(
    #         '[TEST 1] Skipping the ~1hr real nulling_gitl runs below, since '
    #         'they would only be confirming a correction built on an '
    #         'already-broken G. Set FORCE_FULL_RUN=1 to run them anyway.'
    #     )
    #     if os.environ.get('FORCE_FULL_RUN') != '1':
    #         return

    # --- TEST 2 SETUP: build the sine-ripple actuation `a`
    rows, cols = np.meshgrid(np.arange(nact), np.arange(nact), indexing='ij')
    h = RIPPLE_AMP_RAD * np.sin(2 * np.pi * (RIPPLE_FX * rows + RIPPLE_FY * cols) / nact)

    plt.figure()
    im = plt.imshow(h, origin='lower', cmap='RdBu_r')
    plt.colorbar(im, label='DM1 height perturbation a [rad]')
    plt.title(f'Sine-ripple DM actuation (amp={RIPPLE_AMP_RAD} rad, '
              f'fx={RIPPLE_FX}, fy={RIPPLE_FY})')
    plt.savefig(os.path.join(SCRATCH_ROOT, 'checkpoint1_dm_pattern.png'), dpi=120)
    plt.close()
    print('[checkpoint 1] wrote checkpoint1_dm_pattern.png')

    dm10_height = cfg0.dmlist[0].dmvobj.volts_to_dmh(dm10, lamc)
    dm1v_a = cfg0.dmlist[0].dmvobj.dmh_to_volts(dm10_height + h, lamc)
    dm2v_a = dm20

    # VISUAL CHECK: Compare the "flat" baseline to the perturbed sine-ripple DM   
    E_check0 = compact_efield(cfg0, [dm10, dm20], 0)
    E_check1 = compact_efield(cfg0, [dm1v_a, dm2v_a], 0)
    dh_mask_check = insertinto(cfg0.sl_list[0].dh.e, E_check0.shape).astype(bool)
    dE_check = E_check1 - E_check0

    I0 = np.abs(E_check0) ** 2
    I1 = np.abs(E_check1) ** 2
    dI = np.abs(dE_check) ** 2
    norm = max(I0.max(), I1.max())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, img, title in zip(
        axes, [I0, I1, dI],
        ['baseline |C(a0)|^2 (with FPM)', 'perturbed |C(a0+a)|^2 (with FPM)',
         '|C(a0+a) - C(a0)|^2 (ripple contribution only)'],
    ):
        disp = np.log10(img / norm + 1e-12)
        im = ax.imshow(disp, origin='lower', vmin=-8, vmax=0)
        ax.contour(dh_mask_check, colors='red', linewidths=1)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='log10 normalized intensity')
    plt.tight_layout()
    plt.savefig(os.path.join(SCRATCH_ROOT, 'checkpoint2_speckle_in_dh.png'), dpi=120)
    plt.close()

    dh_med = np.median(dI[dh_mask_check])
    n_elevated = int(np.sum(dI[dh_mask_check] > 5 * dh_med))
    print(f'[checkpoint 2] {n_elevated} / {int(np.sum(dh_mask_check))} DH pixels show the '
          'ripple-difference elevated >5x the DH median difference -- inspect '
          'checkpoint2_speckle_in_dh.png (rightmost panel isolates the ripple\'s '
          'own contribution from the off-axis-star baseline leakage visible in '
          'the first two panels); if not, adjust RIPPLE_FX/RIPPLE_FY at the top '
          'of this file.')

    dm1_a_path = os.path.join(SCRATCH_ROOT, 'dmstart_a_dm1.fits')
    dm2_a_path = os.path.join(SCRATCH_ROOT, 'dmstart_a_dm2.fits')
    pyfits.writeto(dm1_a_path, dm1v_a.astype(np.float32), overwrite=True)
    pyfits.writeto(dm2_a_path, dm2v_a.astype(np.float32), overwrite=True)

    # --- Step 2: two real nulling_gitl() runs ---
    run_dir_baseline = os.path.join(SCRATCH_ROOT, 'run_baseline')
    run_dir_perturbed = os.path.join(SCRATCH_ROOT, 'run_perturbed')

    print('Running baseline nulling_gitl (niter=1)...')
    info0 = build_run(niter=1, dmstartmap_filenames=DEFAULT_DMSTARTMAP_FILENAMES,
                       run_dir=run_dir_baseline)

    print('Running perturbed nulling_gitl (niter=2)...')
    info1 = build_run(niter=2, dmstartmap_filenames=[dm1_a_path, dm2_a_path],
                       run_dir=run_dir_perturbed)

    E_c0 = read_meas_efield(run_dir_baseline, 1)
    E_c1 = read_meas_efield(run_dir_perturbed, 1)
    E_corr = read_meas_efield(run_dir_perturbed, 2)

    dh_mask = insertinto(info0['cfg'].sl_list[0].dh.e, E_c0.shape).astype(bool)

    norm = np.nanmax(np.abs(E_c1[dh_mask]) ** 2)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, field, title in zip(
        axes, [E_c0, E_c1, E_corr],
        ['baseline |E_c0|^2', 'perturbed |E_c1|^2 (a applied)', 'corrected |E_corr|^2'],
    ):
        disp = np.log10(np.abs(field) ** 2 / norm + 1e-12)
        im = ax.imshow(disp, origin='lower', vmin=-6, vmax=0)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRATCH_ROOT, 'checkpoint3_dh_before_after.png'), dpi=120)
    plt.close()
    print('[checkpoint 3] wrote checkpoint3_dh_before_after.png')

    # --- Step 3: reuse G0 (computed once in step 0b, identical arguments to
    # nulling_gitl's own internal call) and compare G @ a against the real
    # measured field change ---
    jac, G = jac0, G0

    # howfsc.precomp.howfsc_precomputation builds its own ijlist internally
    # as range(sum(nact**2 for each DM)) -- the full raw actuator range, in
    # cfg.dmlist order (DM1 then DM2), each DM row-major flattened. No tie
    # exclusion, so `a_vec` is just the matching concatenated flatten.
    a_vec = np.concatenate([h.flatten(), np.zeros(dm2_nact * dm2_nact)])
    print(f'[step 3] jac.shape={jac.shape}, a_vec.shape={a_vec.shape}')
    assert G.shape[0] == a_vec.shape[0], (
        'Jacobian actuator axis does not match a_vec length -- check the '
        'DM flatten-order assumption in this script.'
    )

    dE_c_dh = (E_c1 - E_c0)[dh_mask]
    E_g_dh = G.T @ a_vec

    rel_err = np.linalg.norm(dE_c_dh - E_g_dh) / np.linalg.norm(dE_c_dh)
    print(f'[step 2] ||dE_c - G.a|| / ||dE_c|| = {rel_err:.3f} (tol={TOL_STEP2_RELERR})')
    step2_pass = rel_err < TOL_STEP2_RELERR
    print('[step 2] PASS' if step2_pass else '[step 2] FAIL')

    dE_c_map = np.zeros(E_c0.shape, dtype=complex)
    dE_c_map[dh_mask] = dE_c_dh
    E_g_map = np.zeros(E_c0.shape, dtype=complex)
    E_g_map[dh_mask] = E_g_dh

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    panels = [
        (np.real(dE_c_map), 'Re(dE_c) [measured, corgisim/PROPER]'),
        (np.real(E_g_map), 'Re(G . a) [compact-model prediction]'),
        (np.imag(dE_c_map), 'Im(dE_c) [measured, corgisim/PROPER]'),
        (np.imag(E_g_map), 'Im(G . a) [compact-model prediction]'),
    ]
    vmax = max(np.abs(p[0]).max() for p in panels)
    for ax, (data, title) in zip(axes.flat, panels):
        im = ax.imshow(data, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRATCH_ROOT, 'checkpoint4_dEc_vs_Ga.png'), dpi=120)
    plt.close()
    print('[checkpoint 4] wrote checkpoint4_dEc_vs_Ga.png')

    # --- Step 4: did the real control correction (computed entirely inside
    # the perturbed run's own iteration 1) actually null the field? ---
    e0_rms = np.sqrt(np.mean(np.abs(E_c0[dh_mask]) ** 2))
    e1_rms = np.sqrt(np.mean(np.abs(E_c1[dh_mask]) ** 2))
    ecorr_rms = np.sqrt(np.mean(np.abs(E_corr[dh_mask]) ** 2))
    reduction = e1_rms / ecorr_rms if ecorr_rms > 0 else np.inf
    print(f'[step 4] RMS(E_c0)={e0_rms:.3e}, RMS(E_c1)={e1_rms:.3e}, '
          f'RMS(E_corr)={ecorr_rms:.3e}, reduction factor={reduction:.2f}x '
          f'(tol={TOL_STEP4_REDUCTION}x)')

    if not step2_pass:
        print('[step 4] SKIPPED -- step 2 failed, so the correction below was '
              'computed from an already-shown-wrong G; not meaningful to interpret.')
    else:
        step4_pass = reduction > TOL_STEP4_REDUCTION
        print('[step 4] PASS' if step4_pass else '[step 4] FAIL')


if __name__ == '__main__':
    main()
