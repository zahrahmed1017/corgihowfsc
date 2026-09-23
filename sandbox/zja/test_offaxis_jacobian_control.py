"""
Diagnostic: does the off-axis (SNWC/MSWC) compact-model Jacobian G correctly
predict how the dark-hole field responds to a DM move?

Two independent tests, plus a replot stage that only reads from disk:

  check  Fast (seconds, compact-model only) finite-difference self-check of G.
         Writes check/finite_diff.json and a LaTeX table
         check/finite_diff_table.txt.

  run    Run corgihowfsc.gitl.nulling_gitl() twice: once from a flat DM
         baseline (niter=1) and once from baseline + a known sine-ripple DM
         perturbation `a` (niter=2). Everything (normalization, exposure time,
         estimation, control law) is the production code path, and the results
         are read back from the per-iteration FITS files that
         corgihowfsc/utils/saving_output.py always writes. The Jacobian G, the
         dark-hole mask and the ripple/DM maps are saved next to the runs so
         that plots can be regenerated without re-running anything.

  plot   Load a finished `run` from disk, compute metrics (compare the
         full-propagation dE_c = E_c1 - E_c0 to the compact-model prediction
         G.a; measure how much the control step nulled the ripple) and write
         figures/loop_overview.png, figures/loop_diagnostic.png (intensity, shape
         comparison) and figures/loop_field_diagnostic.png (amplitude and phase
         of the complex field, plus the gain / phase / shape decomposition).

  all    check, then run, then plot.

Usage (--out-dir is optional; check/run/all default to a new
<OUTPUT_ROOT>/offaxis_jac_test_<YYYYmmdd_HHMMSS>/ folder, plot defaults to the
newest existing one; pass --out-dir to point at a specific run):
    python test_offaxis_jacobian_control.py check [--out-dir DIR]
    python test_offaxis_jacobian_control.py run   [--out-dir DIR] [--reuse-runs]
    python test_offaxis_jacobian_control.py plot  [--out-dir DIR] [--tol-relerr X]
    python test_offaxis_jacobian_control.py all   [--out-dir DIR]

Every stage is also an importable function (run_fast_check, run_loop_test,
replot, ...), e.g. from a notebook.

Output layout:
    <out_dir>/
      config.json
      check/    finite_diff.json  finite_diff_table.txt
      loop/     run_baseline/  run_perturbed/  dmstart_a_dm{1,2}.fits
                G.npy  dh_mask.npy  ripple.npz  metrics.json
      figures/  loop_overview.png  loop_diagnostic.png  loop_field_diagnostic.png
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from dataclasses import dataclass, asdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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

# Every new run gets its own <OUTPUT_ROOT>/<RUN_PREFIX>_<YYYYmmdd_HHMMSS>/ folder.
OUTPUT_ROOT = os.environ.get(
    'OFFAXIS_TEST_OUTPUT',
    '/Users/Zahra1/Documents/Stanford/Research/corgi-universe/corgiloop_data',
)
RUN_PREFIX = 'offaxis_jac_test'

# Sine-ripple test actuation `a`: amplitude in [0.1, 1] rad (large enough to
# see clearly, small enough for the linear model to be valid), spatial
# frequency so the resulting speckle lands inside the dark hole (10 lam/D away from on-axis)
RIPPLE_AMP_RAD = 0.2
RIPPLE_FX = 100
RIPPLE_FY = -10

TOL_STEP2_RELERR = 0.70
TOL_STEP4_REDUCTION = 5.0


# ---------------------------------------------------------------------------
# Setup / shared building blocks
# ---------------------------------------------------------------------------

@dataclass
class Setup:
    """Everything read from default_param.yml + the model files, resolved once."""
    mode: str
    dark_hole: str
    probe_shape: str
    precomp: str
    backend_type: str
    starting_contrast: float
    dx_mas: float
    dy_mas: float
    corgi_overrides_base: dict
    crop_params: dict
    default_dmstart_files: list
    modelpath: str
    cfgfile: str
    cstratfile: str
    cfg0: object      # off-axis CoronagraphMode (used for Jacobian + fast check)
    dm10: np.ndarray  # baseline DM1 voltages
    dm20: np.ndarray  # baseline DM2 voltages
    nact: int
    dm2_nact: int
    lamc: float


def load_setup():
    """Read default_param.yml, resolve model files, and build the off-axis cfg
    and baseline DM maps."""
    params = loadyaml(PARAM_FILE)
    sim = params['sim_settings']
    model_cfg = params['models'][params['active_model']]
    crop_cfg = params['crop']

    overrides = model_cfg.get('corgi_overrides', {}).copy()
    src = overrides['point_sources'][0]
    dx_mas, dy_mas = src['position_x_mas'], src['position_y_mas']
    default_dmstart = list(model_cfg['dmstartmap_filenames'])

    probe_args = get_args(
        niter=1, mode=sim['mode'], dark_hole=sim['dark_hole'],
        probe_shape=sim['probe_shape'], precomp=sim['precomp'],
        dmstartmap_filenames=default_dmstart,
        fileout=os.path.join(OUTPUT_ROOT, 'unused.fits'),
    )
    modelpath, cfgfile, _, cstratfile, _, _, _, dmstartmaps = load_files(probe_args, HOWFSCPATH)
    dm10, dm20 = dmstartmaps

    cfg0 = build_offaxis_cfg(CoronagraphMode(cfgfile), dx_mas, dy_mas)

    return Setup(
        mode=sim['mode'], dark_hole=sim['dark_hole'],
        probe_shape=sim['probe_shape'], precomp=sim['precomp'],
        backend_type=model_cfg['backend_type'],
        starting_contrast=float(model_cfg['starting_contrast']),
        dx_mas=dx_mas, dy_mas=dy_mas,
        corgi_overrides_base=overrides,
        crop_params={
            'nrow': crop_cfg['nrow'], 'ncol': crop_cfg['ncol'],
            'lrow': model_cfg['lrow'], 'lcol': model_cfg['lcol'],
        },
        default_dmstart_files=default_dmstart,
        modelpath=modelpath, cfgfile=cfgfile, cstratfile=cstratfile,
        cfg0=cfg0, dm10=dm10, dm20=dm20,
        nact=cfg0.dmlist[0].registration['nact'],
        dm2_nact=cfg0.dmlist[1].registration['nact'],
        lamc=float(np.median([sl.lam for sl in cfg0.sl_list])),
    )


def compute_jacobian(setup):
    """Baseline compact-model Jacobian for the off-axis cfg.

    Returns (G, dh_mask): G is complex with shape (n_actuators, n_dh_pixels),
    ordered DM1 then DM2, each row-major flattened (howfsc_precomputation
    builds its own ijlist as range(sum(nact**2 per DM))); dh_mask is the
    boolean dark-hole mask on the (nrow, ncol) crop.
    """
    cp = setup.crop_params
    subcroplist = [(cp['lrow'], cp['lcol'], cp['nrow'], cp['ncol'])]
    jac, _, _ = howfsc_precomputation(
        cfg=setup.cfg0, dmset_list=[setup.dm10, setup.dm20],
        cstrat=ControlStrategy(setup.cstratfile),
        subcroplist=subcroplist, jacmethod='fast', do_n2clist=False,
    )
    G = jac[0] + 1j * jac[1]
    dh_mask = insertinto(setup.cfg0.sl_list[0].dh.e, (cp['nrow'], cp['ncol'])).astype(bool)
    return G, dh_mask


@dataclass
class Ripple:
    """Known sine-ripple DM perturbation `a` applied on top of the baseline."""
    amp: float
    fx: float
    fy: float
    h: np.ndarray        # DM1 height perturbation [rad], (nact, nact)
    dm1v_base: np.ndarray
    dm1v_a: np.ndarray   # baseline + ripple, DM1 voltages
    dm2v: np.ndarray     # DM2 is unchanged
    a_vec: np.ndarray    # h.flatten() then zeros for DM2, matches G's actuator axis


def make_ripple(setup, amp=RIPPLE_AMP_RAD, fx=RIPPLE_FX, fy=RIPPLE_FY):
    nact = setup.nact
    rows, cols = np.meshgrid(np.arange(nact), np.arange(nact), indexing='ij')
    h = amp * np.sin(2 * np.pi * (fx * rows + fy * cols) / nact)

    dm1 = setup.cfg0.dmlist[0].dmvobj
    dm10_height = dm1.volts_to_dmh(setup.dm10, setup.lamc)
    dm1v_a = dm1.dmh_to_volts(dm10_height + h, setup.lamc)
    a_vec = np.concatenate([h.flatten(), np.zeros(setup.dm2_nact ** 2)])
    return Ripple(amp=amp, fx=fx, fy=fy, h=h, dm1v_base=setup.dm10,
                  dm1v_a=dm1v_a, dm2v=setup.dm20, a_vec=a_vec)


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def _read_json(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Stage 1: fast finite-difference check of G
# ---------------------------------------------------------------------------

@dataclass
class FDRow:
    idx: int
    row: int
    col: int
    fd_norm: float        # ||finite-diff of compact-model efield||
    full_G_norm: float    # ||G column||
    raw_edhv_norm: float  # ||pokeprop edhv|| (before epk0/peak term)
    ratio_full: float     # fd_norm / full_G_norm
    corr_full: float      # |<G col, fd>| / (norms)
    ratio_raw: float
    corr_raw: float
    pass_full: bool
    pass_raw: bool


def finite_diff_selfcheck(setup, G, dh_mask, n_test_actuators=5, delta=0.01, corr_tol=0.9):
    """
    Compact-model-only sanity check (no corgisim). For each of the
    n_test_actuators DM1 actuators with the largest Jacobian column norm,
    compares a direct single-actuator finite-difference derivative of the
    compact model's own nonlinear propagation (howfsc.util.prop_tools.efield)
    against TWO analytic quantities:

    1. "full G" -- the Jacobian column jac_solve/inv_to_dm actually use. NOT
       strictly apples-to-apples: per howfsc.control.calcjacs.calcjacs_sp, G's
       column is edhv/epk0 - edhv0/epk0**2*epk, a "joint DM + peak/Strehl"
       Jacobian, while the finite-difference is of the raw field only.
    2. "raw edhv" -- pokeprop()'s direct output (proptolyot/proptodh), taken
       *before* calcjacs_sp applies epk0/epk. Apples-to-apples; isolates
       whether pokeprop/get_jac_precomp are correct, decoupled from the
       peak/Strehl-normalization term. If (1) fails but (2) passes, the fault
       is in epk0/epk (SingleLambda.get_inorm's mean()-based peak extraction,
       which only works for an on-axis field -- see OFFAXIS_JACOBIAN_PLAN.md).

    Returns a list of FDRow.
    """
    cfg, nact, lamc = setup.cfg0, setup.nact, setup.lamc
    dm10, dm20 = setup.dm10, setup.dm20

    col_norms = np.linalg.norm(G[:nact * nact, :], axis=1)
    test_idxs = np.argsort(col_norms)[::-1][:n_test_actuators]

    dm10_height = cfg.dmlist[0].dmvobj.volts_to_dmh(dm10, lamc)
    E0 = compact_efield(cfg, [dm10, dm20], 0)

    sl = cfg.sl_list[0]
    sl.get_jac_precomp([dm10, dm20])  # required precompute for pokeprop()

    rows = []
    for idx in test_idxs:
        r, c = int(idx) // nact, int(idx) % nact
        h_single = np.zeros((nact, nact))
        h_single[r, c] = delta
        dm1v_pert = cfg.dmlist[0].dmvobj.dmh_to_volts(dm10_height + h_single, lamc)
        E1 = compact_efield(cfg, [dm1v_pert, dm20], 0)
        dE_fd = (E1 - E0)[dh_mask] / delta
        fd_norm = np.linalg.norm(dE_fd)

        def _compare(vec):
            n = np.linalg.norm(vec)
            ratio = fd_norm / n if n > 0 else np.inf
            corr = abs(np.vdot(vec, dE_fd)) / (n * fd_norm) if n > 0 and fd_norm > 0 else 0.0
            ok = corr > corr_tol and 0.5 < ratio < 2.0
            return n, ratio, corr, ok

        jac_norm, ratio_full, corr_full, ok_full = _compare(G[idx, :])

        edh = sl.proptodh(sl.proptolyot(sl.pokeprop(int(idx), [dm10, dm20])))
        edhv_norm, ratio_raw, corr_raw, ok_raw = _compare(edh.ravel()[sl.dh_inds])

        rows.append(FDRow(
            idx=int(idx), row=r, col=c, fd_norm=float(fd_norm),
            full_G_norm=float(jac_norm), raw_edhv_norm=float(edhv_norm),
            ratio_full=float(ratio_full), corr_full=float(corr_full),
            ratio_raw=float(ratio_raw), corr_raw=float(corr_raw),
            pass_full=bool(ok_full), pass_raw=bool(ok_raw),
        ))
    return rows


def print_fd_report(rows):
    print('[check] finite-difference self-consistency '
          '(compact-model Jacobian vs compact-model nonlinear propagation):')
    for r in rows:
        print(f'  actuator idx={r.idx} (r={r.row},c={r.col}):')
        print(f'    [1: full G, includes epk0/peak-term] ||jac_col||={r.full_G_norm:.4g}, '
              f'||finite-diff||={r.fd_norm:.4g}, ratio={r.ratio_full:.4g}, '
              f'correlation={r.corr_full:.4g} {"OK" if r.pass_full else "MISMATCH"}')
        print(f'    [2: raw edhv, apples-to-apples]       ||edhv_raw||={r.raw_edhv_norm:.4g}, '
              f'||finite-diff||={r.fd_norm:.4g}, ratio={r.ratio_raw:.4g}, '
              f'correlation={r.corr_raw:.4g} {"OK" if r.pass_raw else "MISMATCH"}')
    print(f'[check] check 1 (full G, what jac_solve actually uses): '
          f'{"PASS" if all(r.pass_full for r in rows) else "FAIL"}')
    print(f'[check] check 2 (raw edhv, apples-to-apples, isolates pokeprop): '
          f'{"PASS" if all(r.pass_raw for r in rows) else "FAIL"}')


def write_fd_table_tex(rows, path):
    """LaTeX tabular (booktabs) with actuator index and the three norms."""
    lines = [
        '% requires \\usepackage{booktabs}',
        '\\begin{tabular}{rrrrrr}',
        '\\toprule',
        'Actuator idx & Row & Col & $\\lVert\\Delta E_{\\mathrm{FD}}\\rVert$ '
        '& $\\lVert G\\rVert$ (full) & $\\lVert E_{\\mathrm{dh}}\\rVert$ (raw) \\\\',
        '\\midrule',
    ]
    for r in rows:
        lines.append(f'{r.idx} & {r.row} & {r.col} & {r.fd_norm:.4e} & '
                     f'{r.full_G_norm:.4e} & {r.raw_edhv_norm:.4e} \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '']
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def run_fast_check(setup, out_dir, G=None, dh_mask=None):
    """Stage 1. Returns the list of FDRow."""
    if G is None:
        print('Computing baseline Jacobian for the self-consistency check...')
        G, dh_mask = compute_jacobian(setup)
    rows = finite_diff_selfcheck(setup, G, dh_mask)
    print_fd_report(rows)

    check_dir = os.path.join(out_dir, 'check')
    _write_json(os.path.join(check_dir, 'finite_diff.json'), [asdict(r) for r in rows])
    write_fd_table_tex(rows, os.path.join(check_dir, 'finite_diff_table.txt'))
    print(f'[check] wrote {check_dir}/finite_diff.json and finite_diff_table.txt')
    return rows


# ---------------------------------------------------------------------------
# Stage 2: the two nulling loops
# ---------------------------------------------------------------------------

def _ensure_preflight_importable():
    """PROPER imports the `roman_preflight` prescription by bare module name,
    so its directory must be on sys.path (also inherited by PROPER's worker
    processes). The production scripts get this by calling
    roman_preflight_proper.copy_here(), which copies the files into the cwd;
    putting the installed package directory on sys.path avoids that clutter."""
    import roman_preflight_proper
    d = os.path.dirname(os.path.abspath(roman_preflight_proper.__file__))
    if d not in sys.path:
        sys.path.insert(0, d)


def run_nulling(setup, niter, dmstart_files, run_dir):
    """Build all objects the same way run_corgisim_nulling_gitl.py::main()
    does, then call the real nulling_gitl() production loop. Output goes to
    run_dir in the usual per-iteration FITS layout."""
    _ensure_preflight_importable()
    os.makedirs(run_dir, exist_ok=True)
    args = get_args(
        niter=niter,
        mode=setup.mode,
        dark_hole=setup.dark_hole,
        probe_shape=setup.probe_shape,
        precomp=setup.precomp,
        num_process=1,
        num_threads=1,
        fileout=os.path.join(run_dir, 'final_frames.fits'),
        jacpath=None,
        dmstartmap_filenames=dmstart_files,
        logfile=os.path.join(run_dir, 'gitl.log'),
    )
    args.starting_contrast = setup.starting_contrast
    args.num_imager_worker = 1
    args.num_proper_process = 1
    args.use_mpi = False
    args.debug = False
    args.mpi_comm = None

    modelpath, cfgfile, jacfile, cstratfile, probefiles, hconffile, \
        n2clistfiles, dmstartmaps = load_files(args, HOWFSCPATH)

    cfg = CoronagraphMode(cfgfile)
    cfg = build_offaxis_cfg(cfg, setup.dx_mas, setup.dy_mas)

    hconf = loadyaml(hconffile, custom_exception=TypeError)
    cstrat = ControlStrategy(cstratfile)
    probes = ProbesShapes(args.probe_shape)

    crop_params = setup.crop_params
    corgi_overrides = setup.corgi_overrides_base.copy()
    corgi_overrides['output_dim'] = crop_params['nrow']
    corgi_overrides = resolve_field_stop_array(corgi_overrides, cfg, modelpath)

    imager = GitlImage(
        cfg=cfg, cstrat=cstrat, hconf=hconf,
        backend=setup.backend_type, cor=setup.mode, corgi_overrides=corgi_overrides,
    )

    # Perfect estimator: meas_efield is then the direct corgisim/PROPER
    # field, not a pairwise-probing reconstruction -- keeps "is G right"
    # decoupled from "does probe-based estimation work". Also reduce probe
    # pairs to 1, matching run_corgisim_nulling_gitl.py's own speedup.
    estimator = PerfectEstimator()
    probefiles = {0: probefiles[0]}
    hconf['probe']['dmrel_ph_list'] = hconf['probe']['dmrel_ph_list'][:1]

    normalization_strategy = EETCNormalization(setup.backend_type, corgi_overrides)
    metadata = {'diagnostic': 'test_offaxis_jacobian_control', 'niter': niter}

    nulling_gitl(
        cstrat, estimator, probes, normalization_strategy, imager,
        cfg, args, hconf, modelpath, jacfile, probefiles, n2clistfiles,
        crop_params, dmstartmaps, metadata=metadata, output_every_iter=True,
    )


def _iter_file(run_dir, iteration_idx, name):
    return os.path.join(run_dir, f'iteration_{iteration_idx:04d}', name)


def read_meas_efield(run_dir, iteration_idx):
    """Read back other[0]['meas_efield'] for a given iteration, as written
    by save_outputs_iter() -> efield_estimations.fits (real/imag interleaved
    planes, one pair per wavelength; nlam=1 here so shape is (2, nrow, ncol))."""
    data = pyfits.getdata(_iter_file(run_dir, iteration_idx, 'efield_estimations.fits'), ext=1)
    return data[0] + 1j * data[1]


def _run_done(run_dir, niter):
    return os.path.exists(_iter_file(run_dir, niter, 'efield_estimations.fits'))


def run_loop_test(setup, out_dir, amp=RIPPLE_AMP_RAD, fx=RIPPLE_FX, fy=RIPPLE_FY,
                  reuse_runs=False, G=None, dh_mask=None):
    """Stage 2: baseline (niter=1) + perturbed (niter=2) nulling loops, and
    save everything replot() needs. Returns the loop directory."""
    loop_dir = os.path.join(out_dir, 'loop')
    os.makedirs(loop_dir, exist_ok=True)

    ripple = make_ripple(setup, amp, fx, fy)
    dm1_a_path = os.path.join(loop_dir, 'dmstart_a_dm1.fits')
    dm2_a_path = os.path.join(loop_dir, 'dmstart_a_dm2.fits')
    pyfits.writeto(dm1_a_path, ripple.dm1v_a.astype(np.float32), overwrite=True)
    pyfits.writeto(dm2_a_path, ripple.dm2v.astype(np.float32), overwrite=True)

    if G is None:
        print('Computing baseline Jacobian...')
        G, dh_mask = compute_jacobian(setup)
    np.save(os.path.join(loop_dir, 'G.npy'), G)
    np.save(os.path.join(loop_dir, 'dh_mask.npy'), dh_mask)
    np.savez(os.path.join(loop_dir, 'ripple.npz'), h=ripple.h, dm1v_base=ripple.dm1v_base,
             dm1v_a=ripple.dm1v_a, dm2v=ripple.dm2v, a_vec=ripple.a_vec)
    _write_json(os.path.join(out_dir, 'config.json'), {
        'ripple_amp_rad': amp, 'ripple_fx': fx, 'ripple_fy': fy,
        'dx_mas': setup.dx_mas, 'dy_mas': setup.dy_mas,
        'mode': setup.mode, 'dark_hole': setup.dark_hole,
        'pixperlod': float(setup.cfg0.sl_list[0].fs.pixperlod),
    })

    run_dir_baseline = os.path.join(loop_dir, 'run_baseline')
    run_dir_perturbed = os.path.join(loop_dir, 'run_perturbed')

    if reuse_runs and _run_done(run_dir_baseline, 1):
        print('Reusing existing baseline run.')
    else:
        print('Running baseline nulling_gitl (niter=1)...')
        run_nulling(setup, 1, setup.default_dmstart_files, run_dir_baseline)

    if reuse_runs and _run_done(run_dir_perturbed, 2):
        print('Reusing existing perturbed run.')
    else:
        print('Running perturbed nulling_gitl (niter=2)...')
        run_nulling(setup, 2, [dm1_a_path, dm2_a_path], run_dir_perturbed)
    return loop_dir


# ---------------------------------------------------------------------------
# Stage 3: load, analyze, plot -- all from disk
# ---------------------------------------------------------------------------

@dataclass
class LoopResults:
    loop_dir: str
    E_c0: np.ndarray      # baseline measured field
    E_c1: np.ndarray      # perturbed measured field (ripple applied)
    E_corr: np.ndarray    # perturbed run after one control step
    dh_mask: np.ndarray
    G: np.ndarray
    ripple: Ripple
    dx_mas: float
    dy_mas: float
    pixperlod: float      # EXCAM pixels per lambda/D (channel 0)


def _loop_dir(out_dir):
    """New layout is <out_dir>/loop/; runs made by the older flat-layout
    script live directly in <out_dir>."""
    d = os.path.join(out_dir, 'loop')
    return d if os.path.isdir(d) else out_dir


def load_loop_results(out_dir):
    """Load a finished `run`. Anything a run from the older (flat-layout)
    script never saved -- G, dh_mask, ripple -- is rebuilt from the model
    (needs load_setup(), slow) and saved so this only happens once."""
    loop_dir = _loop_dir(out_dir)
    cfg_path = os.path.join(out_dir, 'config.json')
    config = _read_json(cfg_path) if os.path.exists(cfg_path) else {}
    amp = config.get('ripple_amp_rad', RIPPLE_AMP_RAD)
    fx = config.get('ripple_fx', RIPPLE_FX)
    fy = config.get('ripple_fy', RIPPLE_FY)

    E_c0 = read_meas_efield(os.path.join(loop_dir, 'run_baseline'), 1)
    E_c1 = read_meas_efield(os.path.join(loop_dir, 'run_perturbed'), 1)
    E_corr = read_meas_efield(os.path.join(loop_dir, 'run_perturbed'), 2)

    g_path = os.path.join(loop_dir, 'G.npy')
    mask_path = os.path.join(loop_dir, 'dh_mask.npy')
    ripple_path = os.path.join(loop_dir, 'ripple.npz')

    pixperlod = config.get('pixperlod')
    setup = None
    if pixperlod is None or not (os.path.exists(g_path) and os.path.exists(mask_path)
                                 and os.path.exists(ripple_path)):
        print('Saved G / dh_mask / ripple / pixel scale not all found; rebuilding from the model...')
        setup = load_setup()
    if pixperlod is None:
        pixperlod = float(setup.cfg0.sl_list[0].fs.pixperlod)
        config['pixperlod'] = pixperlod
        _write_json(cfg_path, config)
    if os.path.exists(g_path) and os.path.exists(mask_path):
        G, dh_mask = np.load(g_path), np.load(mask_path)
    else:
        G, dh_mask = compute_jacobian(setup)
        np.save(g_path, G)
        np.save(mask_path, dh_mask)
    if os.path.exists(ripple_path):
        z = np.load(ripple_path)
        ripple = Ripple(amp=amp, fx=fx, fy=fy, h=z['h'], dm1v_base=z['dm1v_base'],
                        dm1v_a=z['dm1v_a'], dm2v=z['dm2v'], a_vec=z['a_vec'])
    else:
        ripple = make_ripple(setup, amp, fx, fy)
        np.savez(ripple_path, h=ripple.h, dm1v_base=ripple.dm1v_base,
                 dm1v_a=ripple.dm1v_a, dm2v=ripple.dm2v, a_vec=ripple.a_vec)

    assert G.shape[0] == ripple.a_vec.shape[0], (
        'Jacobian actuator axis does not match a_vec length -- check the '
        'DM flatten-order assumption.'
    )
    assert dh_mask.shape == E_c0.shape, (dh_mask.shape, E_c0.shape)

    return LoopResults(loop_dir=loop_dir, E_c0=E_c0, E_c1=E_c1, E_corr=E_corr,
                       dh_mask=dh_mask, G=G, ripple=ripple,
                       dx_mas=config.get('dx_mas', np.nan), dy_mas=config.get('dy_mas', np.nan),
                       pixperlod=pixperlod)


def measured_and_predicted_dE(res):
    """(dE_c, G.a) inside the dark hole, 1-D complex arrays."""
    dE_c = (res.E_c1 - res.E_c0)[res.dh_mask]
    E_g = res.G.T @ res.ripple.a_vec
    return dE_c, E_g


def analyze_loop(res, tol_relerr=TOL_STEP2_RELERR, tol_reduction=TOL_STEP4_REDUCTION):
    """Numbers only. Step 2: ||dE_c - G.a|| / ||dE_c||. Step 4: did the
    control correction inside the perturbed run's iteration 1 null the field."""
    dE_c, E_g = measured_and_predicted_dE(res)
    rel_err = float(np.linalg.norm(dE_c - E_g) / np.linalg.norm(dE_c))

    def rms(E):
        return float(np.sqrt(np.mean(np.abs(E[res.dh_mask]) ** 2)))

    e0, e1, ecorr = rms(res.E_c0), rms(res.E_c1), rms(res.E_corr)
    reduction = e1 / ecorr if ecorr > 0 else float('inf')
    step2_pass = rel_err < tol_relerr
    # best complex scalar alpha with dE_c ~= alpha * G.a: |alpha| and arg(alpha) are
    # the global gain / phase error, the residual after removing it is the shape error
    alpha = np.vdot(E_g, dE_c) / np.vdot(E_g, E_g)
    shape_resid = float(np.linalg.norm(dE_c - alpha * E_g) / np.linalg.norm(dE_c))
    metrics = {
        'step2_rel_err': rel_err, 'step2_tol': tol_relerr, 'step2_pass': bool(step2_pass),
        'alpha_abs': float(abs(alpha)), 'alpha_gain_pred_over_meas': float(1 / abs(alpha)),
        'alpha_phase_deg': float(np.degrees(np.angle(alpha))),
        'rel_err_after_complex_scale': shape_resid,
        'rms_E_c0': e0, 'rms_E_c1': e1, 'rms_E_corr': ecorr,
        'step4_reduction': reduction, 'step4_tol': tol_reduction,
        # step 4 is only meaningful if G itself was shown to be right
        'step4_pass': bool(step2_pass and reduction > tol_reduction),
        'step4_skipped': not step2_pass,
    }
    print(f'[step 2] ||dE_c - G.a|| / ||dE_c|| = {rel_err:.3f} (tol={tol_relerr}) '
          f'{"PASS" if step2_pass else "FAIL"}')
    print(f'[step 2] best fit dE_c ~ alpha*G.a: |alpha|={abs(alpha):.3e} '
          f'(1/|alpha|={1 / abs(alpha):.1f}), arg(alpha)={np.degrees(np.angle(alpha)):.1f} deg, '
          f'residual after removing alpha = {shape_resid:.3f}')
    print(f'[step 4] RMS(E_c0)={e0:.3e}, RMS(E_c1)={e1:.3e}, RMS(E_corr)={ecorr:.3e}, '
          f'reduction={reduction:.2f}x (tol={tol_reduction}x)')
    if not step2_pass:
        print('[step 4] SKIPPED -- step 2 failed, so the correction was computed from '
              'an already-shown-wrong G; not meaningful to interpret.')
    else:
        print(f'[step 4] {"PASS" if metrics["step4_pass"] else "FAIL"}')
    return metrics


def _set_lod_ticks(ax, shape, pixperlod, nbins=6):
    """Relabel an image-plane axes (drawn in pixel coordinates) in lambda/D
    relative to the optical axis, taken as the crop-centre pixel (n // 2).
    Axes are detector (EXCAM) directions: x = column, y = row. Call after the
    axes limits are final."""
    loc = MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10])
    cy, cx = shape[0] // 2, shape[1] // 2
    for axis, center in (('x', cx), ('y', cy)):
        lo, hi = sorted(getattr(ax, f'get_{axis}lim')())
        ticks = loc.tick_values((lo - center) / pixperlod, (hi - center) / pixperlod)
        pix = center + ticks * pixperlod
        keep = (pix >= lo - 1e-9) & (pix <= hi + 1e-9)
        getattr(ax, f'set_{axis}ticks')(pix[keep])
        getattr(ax, f'set_{axis}ticklabels')([f'{t + 0:g}' for t in ticks[keep]])
        getattr(ax, f'set_{axis}label')(rf'{axis} [$\lambda/D$]')


def _log_intensity(field, norm):
    return np.log10(np.abs(field) ** 2 / norm + 1e-12)


def plot_loop_overview(res, path, vmin=-6):
    """(2,3): row 1 = DM1 maps (baseline | baseline+ripple | difference),
    row 2 = image-plane |E|^2 (baseline | baseline+ripple | |E_c1 - E_c0|^2)."""
    rp = res.ripple
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

    # Row 1: DM1 voltage maps, first two share limits, difference is symmetric
    dm_diff = rp.dm1v_a - rp.dm1v_base
    lo = min(rp.dm1v_base.min(), rp.dm1v_a.min())
    hi = max(rp.dm1v_base.max(), rp.dm1v_a.max())
    # a few weak actuators need huge voltages for the same height change; clip
    # the colour scale at the 99th percentile so the ripple itself stays visible
    dmax = np.percentile(np.abs(dm_diff), 99)
    for ax, img, title, kw in [
        (axes[0, 0], rp.dm1v_base, 'DM1 baseline', dict(cmap='viridis', vmin=lo, vmax=hi)),
        (axes[0, 1], rp.dm1v_a, 'DM1 baseline + ripple', dict(cmap='viridis', vmin=lo, vmax=hi)),
        (axes[0, 2], dm_diff, f'DM1 difference (colour clipped at 99th pct; max |dV|={np.abs(dm_diff).max():.0f} V)',
         dict(cmap='RdBu_r', vmin=-dmax, vmax=dmax)),
    ]:
        im = ax.imshow(img, origin='lower', **kw)
        ax.set_title(title)
        ax.set_xlabel('DM1 actuator (column)')
        ax.set_ylabel('DM1 actuator (row)')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='DM1 [V]')

    # Row 2: image-plane intensities, shared log scale normalised to the DH peak
    dE = res.E_c1 - res.E_c0
    norm = max(np.abs(res.E_c0[res.dh_mask]).max() ** 2, np.abs(res.E_c1[res.dh_mask]).max() ** 2)
    for ax, field, title in [
        (axes[1, 0], res.E_c0, r'baseline $|E_{c0}|^2$'),
        (axes[1, 1], res.E_c1, r'baseline + ripple $|E_{c1}|^2$'),
        (axes[1, 2], dE, r'difference $|E_{c1} - E_{c0}|^2$'),
    ]:
        im = ax.imshow(_log_intensity(field, norm), origin='lower', vmin=vmin, vmax=0)
        ax.contour(res.dh_mask, levels=[0.5], colors='red', linewidths=1)
        ax.set_title(title)
        _set_lod_ticks(ax, res.dh_mask.shape, res.pixperlod)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='log10 (I / I_max,DH)')

    star = '' if np.isnan(res.dx_mas) else f'Off-axis star at ({res.dx_mas}, {res.dy_mas}) mas, '
    fig.suptitle(f'{star}sine ripple amp={rp.amp} rad, fx={rp.fx}, fy={rp.fy}')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'wrote {path}')


def plot_loop_diagnostic(res, metrics, path, vmin=-6):
    """|dE_c|^2 from differencing the two nulling-loop fields (full
    propagation) next to |G.a|^2 (compact-model reconstruction)."""
    dE_c, E_g = measured_and_predicted_dE(res)
    dE_c_map = np.zeros(res.E_c0.shape, dtype=complex)
    E_g_map = np.zeros(res.E_c0.shape, dtype=complex)
    dE_c_map[res.dh_mask] = dE_c
    E_g_map[res.dh_mask] = E_g

    outside = ~res.dh_mask
    # zoom to the dark hole (+ margin) since it is a small part of the frame
    rr, cc = np.where(res.dh_mask)
    pad = 5
    r0, r1 = max(rr.min() - pad, 0), min(rr.max() + pad, res.dh_mask.shape[0] - 1)
    c0, c1 = max(cc.min() - pad, 0), min(cc.max() + pad, res.dh_mask.shape[1] - 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for ax, field, title in [
        (axes[0], dE_c_map, r'full propagation: $|E_{c1} - E_{c0}|^2$'),
        (axes[1], E_g_map, r'reconstructed: $|G\,\Delta a|^2$'),
    ]:
        # each panel is normalised to its own peak (shape comparison); the
        # amplitude comparison is in the peak values and the title ratio
        peak = np.abs(field).max() ** 2
        title = f'{title}\npeak = {peak:.2e}'
        disp = np.ma.masked_where(outside, _log_intensity(field, peak))
        im = ax.imshow(disp, origin='lower', vmin=vmin, vmax=0)
        ax.contour(res.dh_mask, levels=[0.5], colors='red', linewidths=1)
        ax.set_xlim(c0, c1)
        ax.set_ylim(r0, r1)
        _set_lod_ticks(ax, res.dh_mask.shape, res.pixperlod)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='log10 (I / own peak)')

    verdict = 'PASS' if metrics['step2_pass'] else 'FAIL'
    fig.suptitle(f"||dE_c - G.a|| / ||dE_c|| = {metrics['step2_rel_err']:.3f} "
                 f"(tol {metrics['step2_tol']}) {verdict};  "
                 f"||G.a|| / ||dE_c|| = {np.linalg.norm(E_g) / np.linalg.norm(dE_c):.1f}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'wrote {path}')


def plot_loop_field_diagnostic(res, metrics, path):
    """(2,3) comparison in complex-field space, which is what G actually acts on.
    Row 1 (amplitude): |dE_c| and |G.a| on one shared log scale, and the ratio
    |G.a| / |dE_c|.  Row 2 (phase): arg dE_c, arg G.a, and the wrapped difference.
    A flat ratio / phase-difference map means only a global gain + rotation
    (normalization- or convention-like); structure there is a real shape error."""
    dE_c, E_g = measured_and_predicted_dE(res)
    shape = res.E_c0.shape
    outside = ~res.dh_mask

    def to_map(v):
        m = np.zeros(shape, dtype=complex)
        m[res.dh_mask] = v
        return np.ma.masked_where(outside, m)

    meas, pred = to_map(dE_c), to_map(E_g)
    ratio_amp = np.ma.masked_where(outside, np.abs(pred) / np.maximum(np.abs(meas), 1e-300))
    phase_diff = np.ma.masked_where(outside, np.degrees(np.angle(pred * np.conj(meas))))

    rr, cc = np.where(res.dh_mask)
    pad = 5
    r0, r1 = max(rr.min() - pad, 0), min(rr.max() + pad, shape[0] - 1)
    c0, c1 = max(cc.min() - pad, 0), min(cc.max() + pad, shape[1] - 1)

    both = np.concatenate([np.abs(dE_c), np.abs(E_g)])
    amp_lo, amp_hi = np.log10(np.percentile(both, 1)), np.log10(both.max())
    lr = np.log10(np.abs(E_g) / np.maximum(np.abs(dE_c), 1e-300))
    lr_lo, lr_hi = np.percentile(lr, [5, 95])

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    panels = [
        (axes[0, 0], np.ma.log10(np.abs(meas)), r'$|E_{c1}-E_{c0}|$ (full propagation)',
         dict(cmap='viridis', vmin=amp_lo, vmax=amp_hi), 'log10 |field| (shared scale)'),
        (axes[0, 1], np.ma.log10(np.abs(pred)), r'$|G\,\Delta a|$ (reconstructed)',
         dict(cmap='viridis', vmin=amp_lo, vmax=amp_hi), 'log10 |field| (shared scale)'),
        (axes[0, 2], np.ma.log10(ratio_amp), r'ratio $|G\,\Delta a| / |E_{c1}-E_{c0}|$',
         dict(cmap='magma', vmin=lr_lo, vmax=lr_hi), 'log10 ratio'),
        (axes[1, 0], np.degrees(np.angle(meas)), r'arg$(E_{c1}-E_{c0})$',
         dict(cmap='twilight', vmin=-180, vmax=180), 'phase [deg]'),
        (axes[1, 1], np.degrees(np.angle(pred)), r'arg$(G\,\Delta a)$',
         dict(cmap='twilight', vmin=-180, vmax=180), 'phase [deg]'),
        (axes[1, 2], phase_diff, r'arg$(G\,\Delta a)$ - arg$(E_{c1}-E_{c0})$',
         dict(cmap='twilight', vmin=-180, vmax=180), 'phase difference [deg]'),
    ]
    for ax, img, title, kw, label in panels:
        im = ax.imshow(img, origin='lower', **kw)
        ax.contour(res.dh_mask, levels=[0.5], colors='red', linewidths=1)
        ax.set_xlim(c0, c1)
        ax.set_ylim(r0, r1)
        _set_lod_ticks(ax, res.dh_mask.shape, res.pixperlod)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)

    # fig.suptitle(
    #     f"best fit dE_c ~ alpha G.a:  1/|alpha| = {metrics['alpha_gain_pred_over_meas']:.1f} "
    #     f"(G.a too large by this factor),  arg(alpha) = {metrics['alpha_phase_deg']:.1f} deg;  "
    #     f"residual after removing alpha = {metrics['rel_err_after_complex_scale']:.3f}  "
    #     f"(before: {metrics['step2_rel_err']:.3f})")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'wrote {path}')


def replot(out_dir, tol_relerr=TOL_STEP2_RELERR, tol_reduction=TOL_STEP4_REDUCTION):
    """Stage 3: regenerate metrics + figures (and the LaTeX table, if the fast
    check was run) purely from files in out_dir. Returns the metrics dict."""
    res = load_loop_results(out_dir)
    metrics = analyze_loop(res, tol_relerr, tol_reduction)
    _write_json(os.path.join(res.loop_dir, 'metrics.json'), metrics)

    fig_dir = os.path.join(out_dir, 'figures')
    plot_loop_overview(res, os.path.join(fig_dir, 'loop_overview.png'))
    plot_loop_diagnostic(res, metrics, os.path.join(fig_dir, 'loop_diagnostic.png'))
    plot_loop_field_diagnostic(res, metrics, os.path.join(fig_dir, 'loop_field_diagnostic.png'))

    fd_json = os.path.join(out_dir, 'check', 'finite_diff.json')
    if os.path.exists(fd_json):
        rows = [FDRow(**d) for d in _read_json(fd_json)]
        write_fd_table_tex(rows, os.path.join(out_dir, 'check', 'finite_diff_table.txt'))
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def new_run_dir():
    """<OUTPUT_ROOT>/<RUN_PREFIX>_<YYYYmmdd_HHMMSS> (not created yet)."""
    return os.path.join(OUTPUT_ROOT, f'{RUN_PREFIX}_{datetime.now():%Y%m%d_%H%M%S}')


def latest_run_dir():
    """Newest existing <RUN_PREFIX>_* folder in OUTPUT_ROOT, or None."""
    dirs = sorted(glob.glob(os.path.join(OUTPUT_ROOT, f'{RUN_PREFIX}_*')))
    dirs = [d for d in dirs if os.path.isdir(d)]
    return dirs[-1] if dirs else None


def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--out-dir', default=None,
                        help=f'output/run directory. Default: a new timestamped folder under '
                             f'{OUTPUT_ROOT} (for `plot`: the newest existing one)')
    run_opts = argparse.ArgumentParser(add_help=False)
    run_opts.add_argument('--amp', type=float, default=RIPPLE_AMP_RAD, help='ripple amplitude [rad]')
    run_opts.add_argument('--fx', type=float, default=RIPPLE_FX)
    run_opts.add_argument('--fy', type=float, default=RIPPLE_FY)
    run_opts.add_argument('--reuse-runs', action='store_true',
                          help='skip a nulling loop if its output already exists in out-dir')
    plot_opts = argparse.ArgumentParser(add_help=False)
    plot_opts.add_argument('--tol-relerr', type=float, default=TOL_STEP2_RELERR)
    plot_opts.add_argument('--tol-reduction', type=float, default=TOL_STEP4_REDUCTION)

    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('check', parents=[common], help='fast finite-difference check of G')
    sub.add_parser('run', parents=[common, run_opts], help='run the baseline + perturbed nulling loops')
    sub.add_parser('plot', parents=[common, plot_opts], help='metrics + figures from a finished run')
    sub.add_parser('all', parents=[common, run_opts, plot_opts], help='check, run, plot')
    args = ap.parse_args(argv)

    if args.cmd == 'plot':
        args.out_dir = args.out_dir or latest_run_dir()
        if args.out_dir is None:
            ap.error(f'no {RUN_PREFIX}_* run found in {OUTPUT_ROOT}; pass --out-dir')
    else:
        if getattr(args, 'reuse_runs', False) and args.out_dir is None:
            ap.error('--reuse-runs needs --out-dir (a fresh timestamped folder has nothing to reuse)')
        args.out_dir = args.out_dir or new_run_dir()
    print(f'Output directory: {args.out_dir}')
    os.makedirs(args.out_dir, exist_ok=True)

    if args.cmd == 'plot':
        replot(args.out_dir, args.tol_relerr, args.tol_reduction)
        return

    setup = load_setup()
    print(f'Off-axis star at ({setup.dx_mas}, {setup.dy_mas}) mas; '
          f'DM1 nact={setup.nact}, lamc={setup.lamc:.4e} m')

    if args.cmd == 'check':
        run_fast_check(setup, args.out_dir)
    elif args.cmd == 'run':
        run_loop_test(setup, args.out_dir, args.amp, args.fx, args.fy, args.reuse_runs)
    elif args.cmd == 'all':
        print('Computing baseline Jacobian...')
        G, dh_mask = compute_jacobian(setup)
        run_fast_check(setup, args.out_dir, G, dh_mask)
        run_loop_test(setup, args.out_dir, args.amp, args.fx, args.fy, args.reuse_runs, G, dh_mask)
        replot(args.out_dir, args.tol_relerr, args.tol_reduction)


if __name__ == '__main__':
    main()
