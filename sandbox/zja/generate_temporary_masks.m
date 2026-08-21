%% Pad/Generate masks for MSWC
% Folder to add to path for this script to run:
% \\exoplanetshare\labdata\FundedProjects\MSWC\MSWC_ISFM2023\hcitOMC\lib_matlab_ames\misc_utilities
% \\exoplanetshare\labdata\FundedProjects\MSWC\MSWC_ISFM2023\hcitOMC\lib_matlab_ames\resample
% falco-matlab\
%% Re-sampled PROPER MSWC SPM for Compact Model 

N_resampled = 300;
roll_angle = deg2rad(90);

orig_spm_path = "C:/Users/zahmed1/Documents/cgisim_cpp/roman_preflight_proper/preflight_data/spc_20200623_mswc/SPM_SPC-20200623_982_rounded9_gray_rotated.fits";
orig_spm = pupil_generate(orig_spm_path, 1, 'full-grid', 'pixel-centered');
resampled_spm = ames_resampleRotatePupil(orig_spm_path, N_resampled, roll_angle);
padded_spm = pad_crop(resampled_spm, 340);

% plot everything
figure();
tiledlayout(1,3,'TileSpacing','compact');
nexttile();
imagesc(orig_spm.A);
axis image; colorbar;
title('Original SPM')

nexttile();
imagesc(resampled_spm);
axis image; colorbar; 
title('Resampled SPM')

nexttile();
imagesc(padded_spm);
axis image; colorbar;
title('Padded SPM')

% Save:
fitswrite(abs(padded_spm), "C:/Users/zahmed1/corgiloop_data/mswc_configuration/spm_mswc_amp.fits")

%% Let's load in the non-mswc spm and perform the same transformation and then compare to the compact model

proper_spm_path = "C:/Users/zahmed1/Documents/cgisim_cpp/roman_preflight_proper/preflight_data/spc_20200610_wfov/SPM_SPC-20200610_1000_rounded9_gray_rotated.fits";
compact_spm_path = "C:/Users/zahmed1/Documents/corgihowfsc/corgihowfsc/model/wfov_band4/any/spm_amp.fits";

orig_spm = pupil_generate(proper_spm_path, 1, 'full-grid', 'pixel-centered');
resampled_spm = ames_resampleRotatePupil(proper_spm_path, N_resampled, roll_angle);
padded_spm = pad_crop(resampled_spm, 340);

compact_spm = pupil_generate(compact_spm_path, 1, 'full-grid', 'pxiel-centered');

% plot everything
figure();
tiledlayout(2,2,'TileSpacing','compact');
nexttile();
imagesc(orig_spm.A);
axis image; colorbar;
title('PROPER SPM')

nexttile();
imagesc(compact_spm.A);
axis image; colorbar; 
title('Compact SPM')

nexttile();
imagesc(padded_spm);
axis image; colorbar;
title('Padded & Resized PROPER SPM')

nexttile();
imagesc(padded_spm - compact_spm.A);
axis image; colorbar;
title("Difference")

%% Square field stop