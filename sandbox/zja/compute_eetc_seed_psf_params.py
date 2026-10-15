"""
Compute peak_flux_ratio_pix, num_pixels and fraction for an EETC sequence entry
from an unocculted corgisim PSF (FPM out, field stop out, SPM + Lyot in), using the
same eetc.sequence_tools helpers that produced the values in sequences.yaml.

The 4a cases are a sanity check against the existing WFOV_WFOV_4A_SEED entry
(peak_flux_ratio_pix 0.022133, num_pixels 8.0, fraction 0.134).

Run from the corgihowfsc repo root:
    python sandbox/zja/compute_eetc_seed_psf_params.py
"""
import os

from astropy.io import fits
from corgisim import instrument, scene
from eetc.sequence_tools import get_num_pixels_and_fraction, get_peak_flux_ratio_pix

MODEL_ANY = 'corgihowfsc/model/wfov_mswc_band4a/any'
OUTPUT_DIM = 201

CASES = [
    ('spc-wide', '4a'),   # reproduces the existing WFOV_WFOV_4A_SEED setup
    ('spc-mswc', '4a'),
    ('spc-mswc', '3d'),
]


def unocculted_psf(cor_type, bandpass, dm1v, dm2v):
    optics_keywords = {
        'cor_type': cor_type,
        'use_errors': 2,
        'polaxis': 10,
        'output_dim': OUTPUT_DIM,
        'use_dm1': 1,
        'dm1_v': dm1v,
        'use_dm2': 1,
        'dm2_v': dm2v,
        'use_fpm': 0,
        'use_lyot_stop': 1,
        'use_field_stop': 0,
        'use_pupil_mask': 1,
    }
    optics = instrument.CorgiOptics('excam', bandpass, optics_keywords=optics_keywords, if_quiet=True)
    host = {'Vmag': 5.0, 'spectral_type': 'G0V', 'magtype': 'vegamag', 'ref_flag': False}
    return optics.get_host_star_psf(scene.Scene(host, [])).host_star_image.data


def main():
    dm1v = fits.getdata(os.path.join(MODEL_ANY, 'dmabs_init_dm1.fits'))
    dm2v = fits.getdata(os.path.join(MODEL_ANY, 'dmabs_init_dm2.fits'))
    for cor_type, bandpass in CASES:
        im = unocculted_psf(cor_type, bandpass, dm1v, dm2v)
        peak = get_peak_flux_ratio_pix(im)
        num_pixels, fraction, _ = get_num_pixels_and_fraction(im, 0.5)
        print(f'{cor_type:9s} {bandpass}: peak_flux_ratio_pix={peak:.6g}  '
              f'num_pixels={float(num_pixels):.6g}  fraction={float(fraction):.6g}')


if __name__ == '__main__':
    main()
