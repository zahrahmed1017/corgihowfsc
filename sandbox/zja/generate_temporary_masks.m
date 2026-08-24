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

%% Square field stop and dark zone

nPix   = 153;                  % grid size (rows = cols), matches EXCAM crop and existing fs/dh_masks
ppl    = 3.1382097356516145;   % pixels per lambda/D for the fs. copy/pasted from howfsc_optical_model.yaml

width_m = 309.1e-6; % width of the square field stop in meters from Table 3 of Riggs et al. March 2025
width_lamD = 9; % width of the square field stop in lamD from Table 3 of Riggs et al. March 2025

% Field stop (square with filleted corners), half-widths in lambda/D
% (full width = 2*halfWidth)
fs_halfWidth_x_lamD = 4.5;    % From Table 3 of AJ Riggs "Flight Masks of the Roman Space Telescope Coronagraph Instrument" Paper
fs_halfWidth_y_lamD = 4.5;
fs_filletRadius_m = 0; %30e-6;    % From Table 3 of Riggs et al. March 2025 (30 micron) 
fs_filletRadius_lamD = convert_fillet_radius(fs_filletRadius_m, width_m, width_lamD);


fs_xOffset_lamD      = 10.0;    % shift the field stop off array-center
fs_yOffset_lamD      = -10.0;
fs_rotDeg            = 0.0;    % clocking of the square stop, if any

% Dark hole (square here; keep annulus option below for reference/blend)
dh_halfWidth_x_lamD  = 2.5;   % Want a 5 lam/D x 5 lam/D dark hole
dh_halfWidth_y_lamD  = 2.5;
dh_xOffset_lamD      = 10.0;   % should generally match/sit inside the field stop
dh_yOffset_lamD      = -10.0;

% edgeSoftnessPix = 1.0;         % width, in pixels, of the smoothed transition at
% the mask edge (analytic, not supersampled).
edgeSoftnessPix = 0.0;
% Set to 0 for a hard binary edge.

outDir = 'sandbox/zja';

%% Field Stop
fs_mask = build_rounded_square_mask(nPix, ppl, fs_halfWidth_x_lamD, fs_halfWidth_y_lamD, ...
                            fs_filletRadius_lamD, fs_xOffset_lamD, fs_yOffset_lamD, ...
                            fs_rotDeg, edgeSoftnessPix);



%% Dark Hole Mask
dh_mask = build_rounded_square_mask(nPix, ppl, dh_halfWidth_x_lamD, dh_halfWidth_y_lamD, ...
                            0.0, dh_xOffset_lamD, dh_yOffset_lamD, 0.0, 0.0);
% DH mask is a hard binary mask
% dh_mask = double(dh_mask > 0.5); % just as a safety check, but I don't
% actually need this

%% Plot with lambda/D axes
% Same center convention as build_rounded_square_mask: c = (nPix+1)/2
c = (nPix + 1) / 2;
lamD_axis = ((1:nPix) - c) / ppl;

figure();
tiledlayout(1,3,'TileSpacing','compact');

nexttile();
imagesc(lamD_axis, lamD_axis, fs_mask);
axis image; axis xy; colorbar;
xlabel('\lambda/D'); ylabel('\lambda/D');
title('Field Stop')

nexttile();
imagesc(lamD_axis, lamD_axis, dh_mask);
axis image; axis xy; colorbar;
xlabel('\lambda/D'); ylabel('\lambda/D');
title('Dark Hole Mask')

nexttile();
imagesc(lamD_axis, lamD_axis, fs_mask);
axis image; axis xy; colorbar;
hold on;
contour(lamD_axis, lamD_axis, dh_mask, [0.5 0.5], 'r', 'LineWidth', 1.5);
hold off;
xlabel('\lambda/D'); ylabel('\lambda/D');
title('Field Stop with DH outline')

%% Save

fs_out = fullfile(outDir, 'fs_amp.fits');
write_fits(fs_out, fs_mask);
fprintf('Wrote field stop amplitude mask: %s\n', fs_out);

dh_fits_out = fullfile(outDir, 'dh_sw_mask.fits');
write_fits(dh_fits_out, dh_mask);
fprintf('Wrote DH mask: %s\n', dh_fits_out);

dh_yaml_out = fullfile(outDir, 'dh_sw_mask.yaml');
write_dh_yaml(dh_yaml_out, nPix, nPix, ppl, ...
    dh_halfWidth_x_lamD, dh_halfWidth_y_lamD, ...
    dh_xOffset_lamD, dh_yOffset_lamD);
fprintf('Wrote DH yaml (documentation only, not read by howfsc): %s\n', dh_yaml_out);

% Also adding the pixelweights fits file here:
fitswrite(ones(153,153), fullfile(outDir,'pixelweights_ones_nlam1_nrow153.fits'));

%% Helper functions
function mask = build_rounded_square_mask(nPix, ppl, halfWidthX_lamD, halfWidthY_lamD, ...
                                   filletRadius_lamD, xOffset_lamD, yOffset_lamD, ...
                                   rotDeg, edgeSoftnessPix)
% Build an nPix x nPix square with fillet corners
% transmission mask, via a signed-distance-function (SDF) test, centered
% at array-center + (xOffset, yOffset) [lambda/D], half-widths
% halfWidthX/Y [lambda/D], fillet radius filletRadius_lamD [lambda/D],
% optional clocking rotDeg, and an analytic anti-aliased edge of width
% edgeSoftnessPix pixels (0 = hard binary edge).
%
% Convention: array center for an odd nPix is at index (nPix+1)/2 (1-indexed),
% matching howfsc's centered-array assumption (ModelElement docstring).
%
% SDF for a rounded rect with half-extents (hx,hy) and corner radius r:
%   qx = |x| - (hx - r);  qy = |y| - (hy - r)
%   d  = hypot(max(qx,0), max(qy,0)) + min(max(qx,qy), 0) - r
% d <= 0 is inside. This reduces to a plain rectangle when r = 0.

if mod(nPix, 2) == 0
    warning('nPix is even; howfsc convention assumes an odd, centered grid.');
end
c = (nPix + 1) / 2;
[X, Y] = meshgrid(1:nPix, 1:nPix);
X = X - c;
Y = Y - c;

xOffset_pix = xOffset_lamD * ppl;
yOffset_pix = yOffset_lamD * ppl;
hx = halfWidthX_lamD * ppl;
hy = halfWidthY_lamD * ppl;
r  = filletRadius_lamD * ppl;
r  = min(r, min(hx, hy));   % clamp: fillet can't exceed the half-width

Xs = X - xOffset_pix;
Ys = Y - yOffset_pix;

if rotDeg ~= 0
    th = deg2rad(rotDeg);
    Xr =  Xs*cos(th) + Ys*sin(th);
    Yr = -Xs*sin(th) + Ys*cos(th);
    Xs = Xr; Ys = Yr;
end

qx = abs(Xs) - (hx - r);
qy = abs(Ys) - (hy - r);
d = hypot(max(qx, 0), max(qy, 0)) + min(max(qx, qy), 0) - r;

if edgeSoftnessPix > 0
    % smoothstep over [-edgeSoftnessPix/2, +edgeSoftnessPix/2] around d=0
    t = min(max(0.5 - d/edgeSoftnessPix, 0), 1);
    mask = t.^2 .* (3 - 2*t);
else
    mask = double(d <= 0);
end

end


function filletRadius_lamD = convert_fillet_radius(filletRadius_m, width_m, width_lamD)
% convert filled radius from meters to lamD
metersPerLamD = width_m / width_lamD;
filletRadius_lamD = filletRadius_m / metersPerLamD;
end


function write_fits(filename, data)
if exist(filename, 'file')
    delete(filename);
end
fitswrite(single(data), filename);
end


function write_dh_yaml(filename, nRows, nCols, ppl, halfWidthX_lamD, halfWidthY_lamD, xOffset_lamD, yOffset_lamD)
% There is a dh_sw_mask.yaml file in each subband in the model folders. I
% can't find anywhere where that yaml is acutally read/used but creating
% one just in case/good for documentation purposes too.
fid = fopen(filename, 'w');
fprintf(fid, 'nRows: %d\n', nRows);
fprintf(fid, 'nCols: %d\n', nCols);
fprintf(fid, 'shapes:\n');
fprintf(fid, '  shape0:\n');
fprintf(fid, '    shape: square\n');
fprintf(fid, '    halfWidthXLamD: %.6f\n', halfWidthX_lamD);
fprintf(fid, '    halfWidthYLamD: %.6f\n', halfWidthY_lamD);
fprintf(fid, '    xOffsetLamD: %.6f\n', xOffset_lamD);
fprintf(fid, '    yOffsetLamD: %.6f\n', yOffset_lamD);
fprintf(fid, '    ppl: %.10f\n', ppl);
fprintf(fid, '    halfWidthXPix: %.6f\n', halfWidthX_lamD * ppl);
fprintf(fid, '    halfWidthYPix: %.6f\n', halfWidthY_lamD * ppl);
fprintf(fid, '    xOffsetPix: %.6f\n', xOffset_lamD * ppl);
fprintf(fid, '    yOffsetPix: %.6f\n', yOffset_lamD * ppl);
fclose(fid);
end
