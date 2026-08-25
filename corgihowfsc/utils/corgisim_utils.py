import os

import numpy as np
from astropy.io import fits

# Mapping configuration - easy to update for new modes
# NOTE - Add new mappings here as support is added

_MANAGER_KEYS = frozenset({
    'bandpass',
    'is_noise_free',
    'Vmag',
    'sptype',
    'ref_flag',
    'point_sources',
    'nlam',
    'host_star_enabled'
})

CGI_TO_CORGI_MAPPING = {
    'narrowfov': 'hlc',
    'nfov_flat': 'hlc', 
    'nfov_dm': 'hlc',
    'nfov_band1': 'hlc',
    'spec_band2': 'spc-spec_band2', 
    'spec_band3': 'spc-spec_band3',
    'wfov_band4': 'spc-wide',
    'wfov_mswc_band4a': 'spc-mswc'
}

SUPPORTED_CGI_MODES = list(CGI_TO_CORGI_MAPPING.keys())
SUPPORTED_CORGI_MODES = list(set(CGI_TO_CORGI_MAPPING.values()))

def _extract_host_properties_from_hconf(hconf):
    """Extract host star properties from hconf object"""
    try:
        star_config = hconf.get('star', {}) if isinstance(hconf, dict) else getattr(hconf, 'star', {})
        
        # Extract stellar properties, preferring target values if available
        Vmag = star_config.get('stellar_vmag')

        sptype = star_config.get('stellar_type')

        return {
            'Vmag': Vmag,
            'spectral_type': sptype,
            'magtype': 'vegamag',  # standard default
            'ref_flag': False  # standard default
        }
    except (AttributeError, KeyError) as e:
        raise ValueError(f"hconf missing required star configuration: {e}")

def _build_point_source_info(point_sources_cfg):
    """ Convert corgi_overrides['point_sources'] into the point_source_info that is expected by corgisim.scene.Scene
    
        Each entry in point_sources_cfg should have:
            - vmag: Vega magnitude 
            - posoition_x_mas: dRA offset from host star in mas
            - poisition_y_mas: dDec offset from host star in mas
    """
    return [{
        'Vmag': src['vmag'], 
        'magtype': 'vegamag', 
        'position_x': src['position_x_mas'], 
        'position_y': src['position_y_mas']
        } for src in point_sources_cfg]
    
# Helper function to map wavelength to corgisim bandpass
def map_wavelength_to_corgisim_bandpass(wavelength_m, tolerance=5e-9):
    """
    Map wavelength to CorgiSim bandpass label.
    
    Args:
        wavelength_m: Wavelength in meters
        tolerance: Matching tolerance in meters (default ±5nm)
        
    Returns:
        CorgiSim bandpass label ('1', '2', '3', or '4')
    """

    corgisim_wavelengths = {
        '1': 575e-9, '2': 660e-9, '3': 730e-9, '4': 825e-9}
    
    for bandpass, wl in corgisim_wavelengths.items():
        if abs(wavelength_m - wl) <= tolerance:
            return bandpass
    
    available_nm = [wl * 1e9 for wl in corgisim_wavelengths.values()]
    raise ValueError(f"Wavelength {wavelength_m*1e9:.1f} nm does not match any "
                    f"CorgiSim options {available_nm} nm within ±{tolerance*1e9:.0f} nm")

def resolve_field_stop_array(corgi_overrides, cfg, modelpath, sl_index=0):
    """
    Convert a field stop fits file into a 2D array that can be passed into PROPER.

    Extract 'field_stop_array_fn' / 'field_stop_width_m' / 'field_stop_width_lamD'
    out of corgi_overrides (does nothing if 'field_stop_array_fn' isn't present) and
    replaces them with 'field_stop_array' (2D np array) and
    'field_stop_array_sampling_m' (meters/pixel at the FSAM plane). The pixel
    scale is derived from the compact model's own fs.ppl for the given
    subband (cfg.sl_list[sl_index].fs.pixperlod), so the full model can't
    drift out of sync with howfsc_optical_model.yaml.

    Call this after cfg = CoronagraphMode(cfgfile) and before the
    corgi_overrides dict is handed to GitlImage (or shipped to MPI workers).

    Args:
        corgi_overrides: dict loaded from default_param.yml, mutated in place
            and also returned for convenience.
        cfg: howfsc.model.mode.CoronagraphMode, already loaded, giving access
            to cfg.sl_list[i].fs.pixperlod.
        modelpath: directory that 'field_stop_array_fn' is relative to (same
            convention as dmstartmap_filenames).
        sl_index: which cfg.sl_list subband's fs.ppl to use. Defaults to 0
            (the only subband for wfov_mswc_band4a).
    """
    fn = corgi_overrides.pop('field_stop_array_fn', None)
    if fn is None:
        return corgi_overrides

    if not os.path.isabs(fn):
        fn = os.path.join(modelpath, fn)

    width_m = corgi_overrides.pop('field_stop_width_m')
    width_lamD = corgi_overrides.pop('field_stop_width_lamD')

    ppl = cfg.sl_list[sl_index].fs.pixperlod
    meters_per_pixel = width_m / (width_lamD * ppl)

    corgi_overrides['field_stop_array'] = fits.getdata(fn).astype(np.float64)
    corgi_overrides['field_stop_array_sampling_m'] = meters_per_pixel

    return corgi_overrides


def calculate_mas_per_lamD(wavelength_m):
    """
    Calculate milliarcseconds per λ/D.
    
    Args:
        wavelength_m: Wavelength in meters
        telescope_diameter_m: Telescope diameter in meters

    """
    D = 2.363114 # Telescope diameter in meters
    theta_rad = wavelength_m / D  # radians
    theta_mas = theta_rad * (180/np.pi) * 3600 * 1000  # convert to mas
    
    return theta_mas