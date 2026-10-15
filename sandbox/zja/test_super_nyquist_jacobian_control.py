"""
Diagnostic: does the off-axis (SNWC/MSWC) compact-model Jacobian G correctly
predict the REAL EFC-computed DM correction, at separations too large for a
DM-actuator sine-ripple to probe directly?

Why this script exists (vs. test_offaxis_jacobian_control.py): that script
validates G by injecting a known sine-ripple DM perturbation tuned to land a
test speckle at a specific lambda/D offset. A nact x nact DM can only
represent ripple spatial frequencies up to ~nact/2 cycles/aperture, so there
is no ripple that places a validation speckle out near the ~100 lam/D
super-Nyquist separations this script targets. test_offaxis_jacobian_control.py
is left as-is (still useful at smaller/sub-Nyquist separations); this script
instead runs a single real nulling_gitl() loop (niter >= 2, no artificial
perturbation) and, for each consecutive iteration pair (i, i+1), recovers the
REAL DM delta the production control law actually applied -- by differencing
the saved dm1_command.fits/dm2_command.fits between iterations (volts),
converting to height via dmvobj.volts_to_dmh -- and checks G against that.

Per iteration_{N:04d}/ in a nulling_gitl run (confirmed from
corgihowfsc/utils/saving_output.py and corgihowfsc/gitl/nulling_gitl.py):
dm1_command.fits/dm2_command.fits hold the absolute DM state (volts) that was
ALREADY ACTIVE and used to collect iteration N's measurement -- i.e. they are
self-consistent with efield_estimations.fits in that same folder. The NEW DM
state computed by iteration N's control law (meant to null iteration N's
measured field) only appears one folder later, in iteration_{N+1:04d}/. So:
  E_i       = efield_estimations.fits in iteration_i      (field measured
              with DM state D_i)
  Da_i      = height(D_{i+1}) - height(D_i)                (the REAL
              correction the control law computed FROM E_i)
  E_{i+1}   = efield_estimations.fits in iteration_{i+1}  (field measured
              with DM state D_i + Da_i)
Four checks per pair, each isolating a different link in the chain
E_i --[control law]--> Da_i --[G]--> predicted dE --[reality]--> E_{i+1}:
  linearity:      G.T @ Da_i          =~ compact_model_dE(Da_i)   (G vs
                  compact model; no corgisim -- isolates G's linearization)
  model_fidelity: compact_model_dE(Da_i) =~ E_{i+1}[dh_mask]-E_i[dh_mask]
                  (compact model vs corgisim; no G -- isolates optical-model
                  fidelity, independent of G or the control law)
  control:        E_i[dh_mask] + G.T @ Da_i =~ 0  (G self-consistency with
                  the real control-law output; no corgisim -- isolates the
                  inversion math, says nothing about whether G is right)
  propagation:    E_i[dh_mask] + G.T @ Da_i =~ E_{i+1}[dh_mask]  (G vs
                  corgisim end-to-end; ~ linearity + model_fidelity combined)
Note PerfectEstimator means E_i IS the true corgisim field by construction,
so there is no "estimation error" to isolate in any of this -- a separate
run with the production probe-based estimator (compared against a
PerfectEstimator run at the same DM states) would be needed for that.

Two independent tests, plus a replot stage that only reads from disk:

  check  Fast (seconds, compact-model only) finite-difference self-check of
         G. Identical in method to test_offaxis_jacobian_control.py's check
         stage -- single-actuator pokes, separation-independent, no ripple
         involved. Writes check/finite_diff.json and a LaTeX table
         check/finite_diff_table.txt.

  run    Run corgihowfsc.gitl.nulling_gitl() once, niter iterations, no
         artificial DM perturbation, with a PerfectEstimator (meas_efield is
         then the direct corgisim/PROPER field, not a pairwise-probing
         reconstruction -- keeps "is G/the control step right" decoupled
         from "does probe-based estimation work"). Results are read back
         from the per-iteration FITS files corgihowfsc/utils/saving_output.py
         always writes. G and the dark-hole mask are saved alongside so
         plots can be regenerated without re-running anything.

  plot   Load a finished `run` from disk, compute the four per-iteration-pair
         checks above and write figures/rms_trend.png plus, per consecutive
         iteration pair, figures/pair_NNNN_NNNN_three_way.png (corgisim vs.
         compact model vs. G.Da -- visualizes linearity/model_fidelity/
         propagation together) and figures/pair_NNNN_NNNN_control_check.png
         (E_i before vs. G-predicted-after -- visualizes the control check).

  all    check, then run, then plot.

Usage (--out-dir is optional; check/run/all default to a new
<OUTPUT_ROOT>/super_nyquist_jac_test_<YYYYmmdd_HHMMSS>/ folder, plot defaults
to the newest existing one; pass --out-dir to point at a specific run):
    python test_super_nyquist_jacobian_control.py check [--out-dir DIR]
    python test_super_nyquist_jacobian_control.py run   [--out-dir DIR] [--niter N] [--reuse-run]
    python test_super_nyquist_jacobian_control.py plot  [--out-dir DIR] [--tol-relerr X]
    python test_super_nyquist_jacobian_control.py all   [--out-dir DIR] [--niter N]

Every stage is also an importable function (run_fast_check, run_real_test,
replot, ...), e.g. from a notebook.

Output layout:
    <out_dir>/
      config.json
      check/    finite_diff.json  finite_diff_table.txt
      run/      iteration_0001/ ...  G.npy  dh_mask.npy
      figures/  rms_trend.png
                pair_0001_0002_three_way.png
                pair_0001_0002_control_check.png
                ...
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
RUN_PREFIX = 'super_nyquist_jac_test'

DEFAULT_NITER = 2
TOL_STEP2_RELERR = 0.30


# ---------------------------------------------------------------------------
# Setup / shared building blocks (identical to test_offaxis_jacobian_control.py)
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
    hconffile: str
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
    modelpath, cfgfile, _, cstratfile, _, hconffile, _, dmstartmaps = load_files(probe_args, HOWFSCPATH)
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
        modelpath=modelpath, cfgfile=cfgfile, cstratfile=cstratfile, hconffile=hconffile,
        cfg0=cfg0, dm10=dm10, dm20=dm20,
        nact=cfg0.dmlist[0].registration['nact'],
        dm2_nact=cfg0.dmlist[1].registration['nact'],
        lamc=float(np.median([sl.lam for sl in cfg0.sl_list])),
    )


def compute_jacobian(setup):
    """Baseline compact-model Jacobian for the off-axis cfg.

    Returns (G, dh_mask, jtwj_map): G is complex with shape (n_actuators,
    n_dh_pixels), ordered DM1 then DM2, each row-major flattened; dh_mask is
    the boolean dark-hole mask on the (nrow, ncol) crop; jtwj_map is unused by
    this script's checks but returned for parity/future use.
    """
    cp = setup.crop_params
    subcroplist = [(cp['lrow'], cp['lcol'], cp['nrow'], cp['ncol'])]
    jac, jtwj_map, _ = howfsc_precomputation(
        cfg=setup.cfg0, dmset_list=[setup.dm10, setup.dm20],
        cstrat=ControlStrategy(setup.cstratfile),
        subcroplist=subcroplist, jacmethod='fast', do_n2clist=False,
    )
    G = jac[0] + 1j * jac[1]
    dh_mask = insertinto(setup.cfg0.sl_list[0].dh.e, (cp['nrow'], cp['ncol'])).astype(bool)
    return G, dh_mask, jtwj_map


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def _read_json(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Stage 1: fast finite-difference check of G (identical method, separation-
# independent -- single-actuator pokes, no ripple)
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
    compact model's own nonlinear propagation against TWO analytic
    quantities:

    1. "full G" -- the Jacobian column jac_solve/inv_to_dm actually use
       (includes the epk0/epk peak-normalization term from calcjacs_sp).
    2. "raw edhv" -- pokeprop()'s direct output, before epk0/epk is applied.
       Apples-to-apples; isolates pokeprop/get_jac_precomp from the
       peak/Strehl-normalization term. If (1) fails but (2) passes, the
       fault is in epk0/epk specifically.

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
        G, dh_mask, _ = compute_jacobian(setup)
    rows = finite_diff_selfcheck(setup, G, dh_mask)
    print_fd_report(rows)

    check_dir = os.path.join(out_dir, 'check')
    _write_json(os.path.join(check_dir, 'finite_diff.json'), [asdict(r) for r in rows])
    write_fd_table_tex(rows, os.path.join(check_dir, 'finite_diff_table.txt'))
    print(f'[check] wrote {check_dir}/finite_diff.json and finite_diff_table.txt')
    return rows


# ---------------------------------------------------------------------------
# Stage 2: a single real nulling loop, no artificial perturbation
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


def run_nulling_real(setup, niter, run_dir):
    """Build all objects the same way run_corgisim_nulling_gitl.py::main()
    does, then call the real nulling_gitl() production loop, starting from
    the yaml's own default DM maps -- no injected ripple/perturbation.
    Output goes to run_dir in the usual per-iteration FITS layout."""
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
        dmstartmap_filenames=setup.default_dmstart_files,
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
    # field, not a pairwise-probing reconstruction -- keeps "is G/the control
    # step right" decoupled from "does probe-based estimation work".
    estimator = PerfectEstimator()
    probefiles = {0: probefiles[0]}
    hconf['probe']['dmrel_ph_list'] = hconf['probe']['dmrel_ph_list'][:1]

    normalization_strategy = EETCNormalization(setup.backend_type, corgi_overrides)
    metadata = {'diagnostic': 'test_super_nyquist_jacobian_control', 'niter': niter}

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


def read_dm_maps(run_dir, iteration_idx):
    """dm1_command.fits/dm2_command.fits: plain single-HDU FITS (ext=0),
    absolute DM voltages (48x48), holding the DM state that was ALREADY
    ACTIVE and used to collect this iteration's measurement -- i.e.
    self-consistent with efield_estimations.fits in the same folder (see
    corgihowfsc/utils/saving_output.py and corgihowfsc/gitl/nulling_gitl.py)."""
    dm1v = pyfits.getdata(_iter_file(run_dir, iteration_idx, 'dm1_command.fits'))
    dm2v = pyfits.getdata(_iter_file(run_dir, iteration_idx, 'dm2_command.fits'))
    return dm1v, dm2v


def _run_done(run_dir, niter):
    return os.path.exists(_iter_file(run_dir, niter, 'efield_estimations.fits'))


def run_real_test(setup, out_dir, niter=DEFAULT_NITER, reuse_run=False, G=None, dh_mask=None):
    """Stage 2: a single real nulling loop (niter iterations, no artificial
    perturbation). Returns the run directory."""
    run_dir = os.path.join(out_dir, 'run')
    os.makedirs(run_dir, exist_ok=True)

    if G is None:
        print('Computing baseline Jacobian...')
        G, dh_mask, _ = compute_jacobian(setup)
    np.save(os.path.join(run_dir, 'G.npy'), G)
    np.save(os.path.join(run_dir, 'dh_mask.npy'), dh_mask)
    _write_json(os.path.join(out_dir, 'config.json'), {
        'niter': niter,
        'dx_mas': setup.dx_mas, 'dy_mas': setup.dy_mas,
        'mode': setup.mode, 'dark_hole': setup.dark_hole,
        'pixperlod': float(setup.cfg0.sl_list[0].fs.pixperlod),
    })

    if reuse_run and _run_done(run_dir, niter):
        print('Reusing existing run.')
    else:
        print(f'Running real nulling_gitl (niter={niter}, no injected perturbation)...')
        run_nulling_real(setup, niter, run_dir)
    return run_dir


# ---------------------------------------------------------------------------
# Stage 3: load, analyze (real DM delta per iteration pair), plot
# ---------------------------------------------------------------------------

@dataclass
class RealRunResults:
    run_dir: str
    niter: int
    E_list: list        # complex fields, index 0 = iteration 1
    dm1v_list: list
    dm2v_list: list
    dh_mask: np.ndarray
    G: np.ndarray
    dx_mas: float
    dy_mas: float
    pixperlod: float


def load_real_run_results(out_dir):
    """Load a finished `run`. Needs config.json (written by run_real_test)
    plus the per-iteration FITS files and the saved G.npy/dh_mask.npy."""
    cfg_path = os.path.join(out_dir, 'config.json')
    config = _read_json(cfg_path)
    niter = config['niter']
    run_dir = os.path.join(out_dir, 'run')

    E_list, dm1v_list, dm2v_list = [], [], []
    for i in range(1, niter + 1):
        E_list.append(read_meas_efield(run_dir, i))
        dm1v, dm2v = read_dm_maps(run_dir, i)
        dm1v_list.append(dm1v)
        dm2v_list.append(dm2v)

    G = np.load(os.path.join(run_dir, 'G.npy'))
    dh_mask = np.load(os.path.join(run_dir, 'dh_mask.npy'))

    assert dh_mask.shape == E_list[0].shape, (dh_mask.shape, E_list[0].shape)

    return RealRunResults(
        run_dir=run_dir, niter=niter, E_list=E_list,
        dm1v_list=dm1v_list, dm2v_list=dm2v_list,
        dh_mask=dh_mask, G=G,
        dx_mas=config.get('dx_mas', np.nan), dy_mas=config.get('dy_mas', np.nan),
        pixperlod=config['pixperlod'],
    )


def dm_delta_vec(setup, dm1v_i, dm2v_i, dm1v_ip1, dm2v_ip1):
    """The REAL DM delta applied between iteration i and i+1, in height
    [rad], flattened/concatenated to match G's actuator axis (DM1 then DM2,
    each row-major flattened) -- the volts-to-height conversion is the same
    one test_offaxis_jacobian_control.py::make_ripple() uses, just applied to
    the production control law's own output instead of a synthetic ripple.
    DM voltage-to-height is nonlinear, so this must difference the two
    heights rather than converting the voltage delta directly."""
    dm1 = setup.cfg0.dmlist[0].dmvobj
    dm2 = setup.cfg0.dmlist[1].dmvobj
    dh1 = dm1.volts_to_dmh(dm1v_ip1, setup.lamc) - dm1.volts_to_dmh(dm1v_i, setup.lamc)
    dh2 = dm2.volts_to_dmh(dm2v_ip1, setup.lamc) - dm2.volts_to_dmh(dm2v_i, setup.lamc)
    return np.concatenate([dh1.flatten(), dh2.flatten()])


def compact_model_dE(setup, dm_i, dm_ip1, dh_mask):
    """Compact model's own FULL NONLINEAR field change for the real DM delta
    between dm_i and dm_ip1, restricted to dh_mask. Computed once per pair in
    analyze_real_run() and shared by check_linearity_real,
    check_model_fidelity_real, and the diagnostic plots (no corgisim here --
    this is purely the compact model propagated at two DM states)."""
    dm1v_i, dm2v_i = dm_i
    dm1v_ip1, dm2v_ip1 = dm_ip1
    E0 = compact_efield(setup.cfg0, [dm1v_i, dm2v_i], 0)
    E1 = compact_efield(setup.cfg0, [dm1v_ip1, dm2v_ip1], 0)
    return (E1 - E0)[dh_mask]


def check_linearity_real(G, delta_vec, dE_compact):
    """Does G's LINEAR extrapolation (G.T @ delta_vec) match the compact
    model's own FULL NONLINEAR response (dE_compact) to the same real DM
    delta? Compact-model only, no corgisim -- isolates "does G's linearity
    hold at this real control-step amplitude" from "does the compact model
    agree with corgisim" (check_model_fidelity_real)."""
    dE_lin = G.T @ delta_vec
    denom = np.linalg.norm(dE_compact)
    rel_err = float(np.linalg.norm(dE_compact - dE_lin) / denom) if denom > 0 else float('inf')
    corr = float(abs(np.vdot(dE_lin, dE_compact)) /
                (np.linalg.norm(dE_lin) * denom)) if denom > 0 else 0.0
    return {'rel_err': rel_err, 'corr': corr}


def check_model_fidelity_real(E_i, E_ip1, dh_mask, dE_compact):
    """Does the compact model's own nonlinear response to the real DM delta
    (dE_compact) match corgisim's real measured response to that SAME delta
    (E_{i+1} - E_i)? No G anywhere in this check -- isolates compact-model-
    vs-corgisim optical-model fidelity, independent of G's linearization
    (check_linearity_real) and independent of the control law
    (check_control_real). check_propagation_real (G vs corgisim) is roughly
    the combination of this check and check_linearity_real."""
    dE_corgisim = (E_ip1 - E_i)[dh_mask]
    denom = np.linalg.norm(dE_corgisim)
    rel_err = float(np.linalg.norm(dE_compact - dE_corgisim) / denom) if denom > 0 else float('inf')
    denom_c = np.vdot(dE_compact, dE_compact)
    alpha = np.vdot(dE_compact, dE_corgisim) / denom_c if denom_c != 0 else 0.0
    shape_resid = float(np.linalg.norm(dE_corgisim - alpha * dE_compact) / denom) if denom > 0 else float('inf')
    return {
        'rel_err': rel_err, 'alpha_abs': float(abs(alpha)),
        'alpha_phase_deg': float(np.degrees(np.angle(alpha))),
        'rel_err_after_complex_scale': shape_resid,
    }


def check_control_real(E_i, G, dh_mask, delta_vec):
    """Does the REAL correction the control law computed from E_i actually
    null E_i, in G's own linear sense? No re-run of jac_solve -- delta_vec is
    whatever the production control law (with its real weighting/
    regularization) actually produced; this checks the combination of (real
    G, real control law, real estimated field) end to end."""
    e0 = E_i[dh_mask]
    e_after = e0 + G.T @ delta_vec
    field_null_ratio = float(np.linalg.norm(e_after) / np.linalg.norm(e0))
    return {'field_null_ratio': field_null_ratio}


def check_propagation_real(E_i, E_ip1, G, dh_mask, delta_vec, tol_relerr=TOL_STEP2_RELERR):
    """Does propagating the real DM correction reproduce the next real
    measurement? Compares the full-propagation (corgisim) field change
    E_{i+1} - E_i to the compact-model prediction G.T @ delta_vec -- the
    direct real-data analogue of test_offaxis_jacobian_control.py's ripple-
    based Step 2, but against the production control law's own correction."""
    dE_meas = (E_ip1 - E_i)[dh_mask]
    dE_pred = G.T @ delta_vec
    denom = np.linalg.norm(dE_meas)
    rel_err = float(np.linalg.norm(dE_meas - dE_pred) / denom) if denom > 0 else float('inf')
    alpha = np.vdot(dE_pred, dE_meas) / np.vdot(dE_pred, dE_pred) if np.vdot(dE_pred, dE_pred) != 0 else 0.0
    shape_resid = float(np.linalg.norm(dE_meas - alpha * dE_pred) / denom) if denom > 0 else float('inf')
    return {
        'rel_err': rel_err, 'tol': tol_relerr, 'pass': bool(rel_err < tol_relerr),
        'alpha_abs': float(abs(alpha)), 'alpha_phase_deg': float(np.degrees(np.angle(alpha))),
        'rel_err_after_complex_scale': shape_resid,
    }


def analyze_real_run(setup, res, tol_relerr=TOL_STEP2_RELERR):
    """Per-consecutive-iteration-pair metrics. Returns (pair_metrics,
    delta_vecs, dE_compacts): pair_metrics is JSON-serializable (for
    metrics.json); delta_vecs/dE_compacts (one np.ndarray per pair) are kept
    separately for plotting.

    Four checks per pair, each isolating a different link in the chain
    E_i --[control law]--> delta_vec --[G]--> predicted dE --[reality]--> E_ip1:
      linearity:      G vs compact model (no corgisim)       -- Jacobian linearization
      model_fidelity: compact model vs corgisim (no G)        -- optical-model fidelity
      control:        G self-consistency with the real delta_vec -- inversion math
      propagation:    G vs corgisim end-to-end (~ linearity + model_fidelity combined)
    """
    pair_metrics, delta_vecs, dE_compacts = [], [], []
    for i in range(res.niter - 1):
        dm_i = (res.dm1v_list[i], res.dm2v_list[i])
        dm_ip1 = (res.dm1v_list[i + 1], res.dm2v_list[i + 1])
        delta_vec = dm_delta_vec(setup, dm_i[0], dm_i[1], dm_ip1[0], dm_ip1[1])
        dE_compact = compact_model_dE(setup, dm_i, dm_ip1, res.dh_mask)

        lin = check_linearity_real(res.G, delta_vec, dE_compact)
        fid = check_model_fidelity_real(res.E_list[i], res.E_list[i + 1], res.dh_mask, dE_compact)
        ctrl = check_control_real(res.E_list[i], res.G, res.dh_mask, delta_vec)
        prop = check_propagation_real(res.E_list[i], res.E_list[i + 1], res.G, res.dh_mask,
                                      delta_vec, tol_relerr)

        rms_i = float(np.sqrt(np.mean(np.abs(res.E_list[i][res.dh_mask]) ** 2)))
        rms_ip1 = float(np.sqrt(np.mean(np.abs(res.E_list[i + 1][res.dh_mask]) ** 2)))
        reduction = rms_i / rms_ip1 if rms_ip1 > 0 else float('inf')

        print(f'--- iteration {i + 1} -> {i + 2} ---')
        print(f'[linearity]      ||G.da - dE_compact|| / ||dE_compact|| = {lin["rel_err"]:.3f}, '
              f'correlation = {lin["corr"]:.4f}   (G vs compact model only, no corgisim)')
        print(f'[model fidelity] ||dE_compact - dE_corgisim|| / ||dE_corgisim|| = {fid["rel_err"]:.3f}; '
              f'|alpha|={fid["alpha_abs"]:.3e}, arg(alpha)={fid["alpha_phase_deg"]:.1f} deg, '
              f'residual after removing alpha = {fid["rel_err_after_complex_scale"]:.3f}   '
              f'(compact model vs corgisim, no G)')
        print(f'[control]        ||E_i + G.da|| / ||E_i|| = {ctrl["field_null_ratio"]:.4f} '
              f'(fraction of field left after the real correction, in G\'s own linear sense; '
              f'should be << 1)')
        print(f'[propagation]    ||dE_corgisim - G.da|| / ||dE_corgisim|| = {prop["rel_err"]:.3f} '
              f'(tol={tol_relerr}) {"PASS" if prop["pass"] else "FAIL"}; '
              f'|alpha|={prop["alpha_abs"]:.3e}, arg(alpha)={prop["alpha_phase_deg"]:.1f} deg, '
              f'residual after removing alpha = {prop["rel_err_after_complex_scale"]:.3f}   '
              f'(G vs corgisim end-to-end)')
        print(f'[reduction]      RMS(E_i)={rms_i:.3e} -> RMS(E_i+1)={rms_ip1:.3e}, '
              f'reduction={reduction:.2f}x')

        pair_metrics.append({
            'iteration_pair': [i + 1, i + 2],
            'linearity': lin, 'model_fidelity': fid, 'control': ctrl, 'propagation': prop,
            'rms_E_i': rms_i, 'rms_E_ip1': rms_ip1, 'reduction': reduction,
        })
        delta_vecs.append(delta_vec)
        dE_compacts.append(dE_compact)
    return pair_metrics, delta_vecs, dE_compacts


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


def plot_rms_trend(res, pair_metrics, path):
    """Dark-hole RMS(|E|) vs iteration, log scale, with the per-pair
    reduction factor annotated -- the quickest visual check of whether the
    real loop is actually nulling at this separation."""
    rms = [float(np.sqrt(np.mean(np.abs(e[res.dh_mask]) ** 2))) for e in res.E_list]
    iters = np.arange(1, res.niter + 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(iters, rms, 'o-')
    for m in pair_metrics:
        i0, i1 = m['iteration_pair']
        ax.annotate(f'{m["reduction"]:.2f}x', xy=(i1, rms[i1 - 1]),
                    xytext=(2, 6), textcoords='offset points', fontsize=8)
    ax.set_xlabel('iteration')
    ax.set_ylabel('RMS(|E|) in dark hole')
    ax.set_title(f'Off-axis star at ({res.dx_mas}, {res.dy_mas}) mas')
    ax.grid(True, which='both', alpha=0.3)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'wrote {path}')


def _dh_zoom_limits(dh_mask, pad=5):
    rr, cc = np.where(dh_mask)
    r0, r1 = max(rr.min() - pad, 0), min(rr.max() + pad, dh_mask.shape[0] - 1)
    c0, c1 = max(cc.min() - pad, 0), min(cc.max() + pad, dh_mask.shape[1] - 1)
    return r0, r1, c0, c1


def plot_three_way_pair_diagnostic(res, i, metrics, delta_vec, dE_compact, path, vmin=-6):
    """Three-way field-CHANGE comparison for one consecutive iteration pair,
    each restricted to the dark hole and shown on its own-peak log-intensity
    scale:

      1. corgisim (ground truth):   E_{i+1} - E_i, the real measurement
      2. compact model (nonlinear): compact_model_dE() for the same real
         delta_vec -- panel 1 vs 2 is check_model_fidelity_real (isolates
         compact-model-vs-corgisim fidelity, no G involved)
      3. G . delta_vec (linear):    the Jacobian's linear prediction --
         panel 2 vs 3 is check_linearity_real (isolates G's linearization
         error from the compact model); panel 1 vs 3 is
         check_propagation_real (the end-to-end total, ~ 2-vs-3 + 1-vs-2)
    """
    E_i, E_ip1 = res.E_list[i], res.E_list[i + 1]
    dE_corgisim = (E_ip1 - E_i)[res.dh_mask]
    dE_lin = res.G.T @ delta_vec

    def to_map(vec):
        m = np.zeros(E_i.shape, dtype=complex)
        m[res.dh_mask] = vec
        return m

    maps = [to_map(dE_corgisim), to_map(dE_compact), to_map(dE_lin)]
    titles = [
        rf'corgisim (truth): $|E_{{{i+2}}}-E_{{{i+1}}}|^2$',
        r'compact model (nonlinear): $|\Delta E_{\mathrm{compact}}|^2$',
        r'$G\,\Delta a$ (linear): $|G\,\Delta a|^2$',
    ]

    outside = ~res.dh_mask
    r0, r1, c0, c1 = _dh_zoom_limits(res.dh_mask)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for ax, field, title in zip(axes, maps, titles):
        peak = np.abs(field).max() ** 2
        disp = np.ma.masked_where(outside, _log_intensity(field, peak))
        im = ax.imshow(disp, origin='lower', vmin=vmin, vmax=0)
        ax.contour(res.dh_mask, levels=[0.5], colors='red', linewidths=1)
        ax.set_xlim(c0, c1)
        ax.set_ylim(r0, r1)
        _set_lod_ticks(ax, res.dh_mask.shape, res.pixperlod)
        ax.set_title(f'{title}\npeak = {peak:.2e}')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='log10 (I / own peak)')

    lin, fid, prop = metrics['linearity'], metrics['model_fidelity'], metrics['propagation']
    fig.suptitle(
        f'iteration {i + 1} -> {i + 2}:   '
        f'model fidelity (1 vs 2) rel_err={fid["rel_err"]:.3f}   |   '
        f'linearity (2 vs 3) rel_err={lin["rel_err"]:.3f}   |   '
        f'propagation (1 vs 3) rel_err={prop["rel_err"]:.3f} (tol {prop["tol"]}) '
        f'{"PASS" if prop["pass"] else "FAIL"}'
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'wrote {path}')


def plot_control_check(res, i, metrics, delta_vec, path, vmin=-6):
    """Visualizes the control check: the real measured field E_i (before
    correction) next to G's own linear prediction of the corrected field,
    E_i + G.delta_vec. The second panel is NOT a real measurement -- it is
    purely what G itself believes the real correction does. A large drop
    from panel 1 to panel 2 means the control law's solve is self-consistent
    GIVEN this G; it says nothing about whether G matches corgisim (that's
    plot_three_way_pair_diagnostic)."""
    E_i = res.E_list[i]
    e0 = E_i[res.dh_mask]
    e_after = e0 + res.G.T @ delta_vec

    def to_map(vec):
        m = np.zeros(E_i.shape, dtype=complex)
        m[res.dh_mask] = vec
        return m

    maps = [to_map(e0), to_map(e_after)]
    titles = [
        r'before: $|E_i|^2$ (real measurement)',
        r'G-predicted after: $|E_i + G\,\Delta a|^2$ (NOT a real measurement)',
    ]

    outside = ~res.dh_mask
    r0, r1, c0, c1 = _dh_zoom_limits(res.dh_mask)
    # shared scale, normalized to the "before" peak, so the drop (or lack of
    # one) between panels is visible
    norm = np.abs(e0).max() ** 2

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for ax, field, title in zip(axes, maps, titles):
        disp = np.ma.masked_where(outside, _log_intensity(field, norm))
        im = ax.imshow(disp, origin='lower', vmin=vmin, vmax=0)
        ax.contour(res.dh_mask, levels=[0.5], colors='red', linewidths=1)
        ax.set_xlim(c0, c1)
        ax.set_ylim(r0, r1)
        _set_lod_ticks(ax, res.dh_mask.shape, res.pixperlod)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='log10 (I / ||E_i||_peak)')

    ctrl = metrics['control']
    fig.suptitle(f'iteration {i + 1} control check: '
                 f"||E_i + G.da|| / ||E_i|| = {ctrl['field_null_ratio']:.4f} "
                 f'(should be << 1; self-consistent within G only, no corgisim)')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'wrote {path}')


def replot(out_dir, tol_relerr=TOL_STEP2_RELERR):
    """Stage 3: regenerate metrics + figures purely from files in out_dir.
    Returns the pair_metrics list."""
    setup = load_setup()
    res = load_real_run_results(out_dir)
    pair_metrics, delta_vecs, dE_compacts = analyze_real_run(setup, res, tol_relerr)
    _write_json(os.path.join(res.run_dir, 'metrics.json'), pair_metrics)

    fig_dir = os.path.join(out_dir, 'figures')
    plot_rms_trend(res, pair_metrics, os.path.join(fig_dir, 'rms_trend.png'))
    for i, (m, dv, dc) in enumerate(zip(pair_metrics, delta_vecs, dE_compacts)):
        i0, i1 = m['iteration_pair']
        plot_three_way_pair_diagnostic(
            res, i, m, dv, dc,
            os.path.join(fig_dir, f'pair_{i0:04d}_{i1:04d}_three_way.png'))
        plot_control_check(
            res, i, m, dv,
            os.path.join(fig_dir, f'pair_{i0:04d}_{i1:04d}_control_check.png'))

    fd_json = os.path.join(out_dir, 'check', 'finite_diff.json')
    if os.path.exists(fd_json):
        rows = [FDRow(**d) for d in _read_json(fd_json)]
        write_fd_table_tex(rows, os.path.join(out_dir, 'check', 'finite_diff_table.txt'))
    return pair_metrics


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
    run_opts.add_argument('--niter', type=int, default=DEFAULT_NITER,
                          help='number of real nulling_gitl iterations to run (needs >= 2 '
                               'to get at least one consecutive-pair comparison)')
    run_opts.add_argument('--reuse-run', action='store_true',
                          help='skip the nulling loop if its output already exists in out-dir')
    plot_opts = argparse.ArgumentParser(add_help=False)
    plot_opts.add_argument('--tol-relerr', type=float, default=TOL_STEP2_RELERR)

    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('check', parents=[common], help='fast finite-difference check of G')
    sub.add_parser('run', parents=[common, run_opts], help='run a single real nulling loop')
    sub.add_parser('plot', parents=[common, plot_opts], help='metrics + figures from a finished run')
    sub.add_parser('all', parents=[common, run_opts, plot_opts], help='check, run, plot')
    args = ap.parse_args(argv)

    if args.cmd == 'plot':
        args.out_dir = args.out_dir or latest_run_dir()
        if args.out_dir is None:
            ap.error(f'no {RUN_PREFIX}_* run found in {OUTPUT_ROOT}; pass --out-dir')
    else:
        if getattr(args, 'reuse_run', False) and args.out_dir is None:
            ap.error('--reuse-run needs --out-dir (a fresh timestamped folder has nothing to reuse)')
        args.out_dir = args.out_dir or new_run_dir()
    print(f'Output directory: {args.out_dir}')
    os.makedirs(args.out_dir, exist_ok=True)

    if args.cmd == 'plot':
        replot(args.out_dir, args.tol_relerr)
        return

    setup = load_setup()
    print(f'Off-axis star at ({setup.dx_mas}, {setup.dy_mas}) mas; '
          f'DM1 nact={setup.nact}, lamc={setup.lamc:.4e} m')

    if args.cmd == 'check':
        run_fast_check(setup, args.out_dir)
    elif args.cmd == 'run':
        run_real_test(setup, args.out_dir, args.niter, args.reuse_run)
    elif args.cmd == 'all':
        print('Computing baseline Jacobian...')
        G, dh_mask, _ = compute_jacobian(setup)
        run_fast_check(setup, args.out_dir, G, dh_mask)
        run_real_test(setup, args.out_dir, args.niter, args.reuse_run, G, dh_mask)
        replot(args.out_dir, args.tol_relerr)


if __name__ == '__main__':
    main()
