% Plot Masks
%{
    Quick reference for MATLAB fits functions:
        - fitsread = extracts data from primary HDU
        - fitsdisp = prints Header info
        - fitsinfo = struct with all of the .fits information
%}
clc; clear;
%% SPM
% PROPER SPM (full-res, no mswc grating)
proper_path     = "/Users/Zahra1/Documents/Stanford/Research/corgi-universe/cgisim_cpp/roman_preflight_proper/preflight_data/spc_20200610_wfov/SPM_SPC-20200610_1000_rounded9_gray_rotated.fits";
% Compact model SPM (downsampled, no mswc grating)
compact_path    = "/Users/Zahra1/Documents/Stanford/Research/corgi-universe/corgihowfsc/corgihowfsc/model/wfov_band4/any/spm_amp.fits";
%PROPER MSWC SPM (full-res)
proper_mswc_path = "/Users/Zahra1/Documents/Stanford/Research/corgi-universe/cgisim_cpp/roman_preflight_proper/preflight_data/spc_20200623_mswc/SPM_SPC-20200623_982_rounded9_gray_rotated.fits";

% fitsdisp(compact_path);

%% EPUP (backing up a bit here):
% PROPER Epup (full-res)
proper_path     = "/Users/Zahra1/Documents/Stanford/Research/corgi-universe/cgisim_cpp/roman_preflight_proper/preflight_data/spc_20200610_wfov/pupil_SPC-20200610_1000.fits";
% Compact model SPM (downsampled, no mswc grating)
compact_path    = "/Users/Zahra1/Documents/Stanford/Research/corgi-universe/corgihowfsc/corgihowfsc/model/wfov_band4/any/epup_amp.fits";
%PROPER Epup (full-res)
proper_mswc_path = "/Users/Zahra1/Documents/Stanford/Research/corgi-universe/cgisim_cpp/roman_preflight_proper/preflight_data/spc_20200623_mswc/pupil_SPC-20200623_982_rotated.fits";

%% Lyot Stop
% PROPER Epup (full-res)
proper_path     = "/Users/Zahra1/Documents/Stanford/Research/corgi-universe/cgisim_cpp/roman_preflight_proper/preflight_data/spc_20200610_wfov/LS_SPC-20200610_1000.fits";
% Compact model SPM (downsampled, no mswc grating)
compact_path    = "/Users/Zahra1/Documents/Stanford/Research/corgi-universe/corgihowfsc/corgihowfsc/model/wfov_band4/any/lyot_amp.fits";
%PROPER Epup (full-res)
proper_mswc_path = "/Users/Zahra1/Documents/Stanford/Research/corgi-universe/cgisim_cpp/roman_preflight_proper/preflight_data/spc_20200623_mswc/LS_SPC-20200623_982_rotated.fits";

%% Load and Plot

proper = fitsread(proper_path);
compact = fitsread(compact_path);
proper_mswc = fitsread(proper_mswc_path);

proper_center = ceil((size(proper)+1)/2);
compact_center = ceil((size(compact)+1)/2);
proper_mswc_center = ceil((size(proper_mswc)+1)/2);

figure();
tiledlayout(3,3,'TileSpacing','compact')

% Full masks
nexttile();
imagesc(proper);
set(gca,'YDir', 'normal')
title('PROPER')
axis image

nexttile();
imagesc(compact);
set(gca,'YDir', 'normal')
title('Compact Model')
axis image

nexttile();
imagesc(proper_mswc);
set(gca,'YDir', 'normal');
title('PROPER MSWC')
axis image

% Horizontal Center Cut
nexttile();
plot(proper(proper_center(1),:), 'LineWidth',2);
xlabel('Horizontal Pixels (Center Cut)')
ylabel('Amplitude')

nexttile();
plot(compact(compact_center(1),:), 'LineWidth',2);
xlabel('Horizontal Pixels (Center Cut)')

nexttile();
plot(proper_mswc(proper_mswc_center(1),:), 'LineWidth',2);
xlabel('Horizontal Pixels (Center Cut)');

% Vertical Cut
nexttile();
plot(proper(:,proper_center(2)), 'LineWidth',2);
xlabel('Vertical Pixels (Center Cut)')
ylabel('Amplitude')

nexttile();
plot(compact(:,compact_center(2)), 'LineWidth',2);
xlabel('Vertical Pixels (Center Cut)')

nexttile();
plot(proper_mswc(:,proper_mswc_center(2)), 'LineWidth',2);
xlabel('Vertical Pixels (Center Cut)');

%% Estimate 50% outer-edge diameter for compact model

% Compact model has aberrations so going to just take the median of the
% horizontal region that looks relatively steady
compact_med_amp = median(compact(compact_center(1), 222:297));
row  = compact(compact_center(1), :);
col  = compact(:,compact_center(2));
half = compact_med_amp/2;

% outer edge: start on the plateau, walk outward
xL = edge50(row, half, 60, 1);
xR = edge50(row, half, 290, numel(row));
xT = edge50(col, half, 60, 1);
xB = edge50(col, half, 290, numel(col));

% central obscuration: start on the plateau, walk toward the centre
xCL = edge50(row, half, 100, compact_center(2));
xCR = edge50(row, half, 240, compact_center(2));
xCT = edge50(col, half, 100, compact_center(1));
xCB = edge50(col, half, 240, compact_center(1));

D_outer = xR - xL;
D_cobs  = xCR - xCL;
D_outer_tb = xB - xT;
D_cobs_tb  = xCB - xCT;

%% Okay now for the PROPER mask (let's do the mswc since that's what we'll be matching to)
row = proper_mswc(proper_mswc_center(1), :);
col = proper_mswc(:, proper_mswc_center(2));

xL_pr = edge50(row, 0.5, 120, 1);
xR_pr = edge50(row, 0.5, 900, numel(row));
xT_pr = edge50(col, 0.5, 200, 1);
xB_pr = edge50(col, 0.5, 800, numel(col));

% central obscuration: start on the plateau, walk toward the centre
xCL_pr = edge50(row, 0.5, 300, proper_center(2));
xCR_pr = edge50(row, 0.5, 700, proper_center(2));
xCT_pr = edge50(col, 0.5, 200, proper_center(1));
xCB_pr = edge50(col, 0.5, 800, proper_center(1));

D_outer_pr = xR_pr - xL_pr;
D_cobs_pr  = xCR_pr - xCL_pr;
D_outer_tb_pr = xB_pr - xT_pr;
D_cobs_tb_pr  = xCB_pr - xCT_pr;


%% 50% edge helper function:

function x = edge50(profile, level, iInside, iOutside)
% Sub-pixel position where PROFILE crosses LEVEL, walking from iInside
% (a point on the open plateau) toward iOutside (a blocked region).
% Uses the LAST open->blocked crossing, so dips in the plateau that
% happen to touch LEVEL are skipped.
step = sign(iOutside - iInside);
idx  = iInside:step:iOutside;
v    = profile(idx);
k = find(v(1:end-1) >= level & v(2:end) < level, 1, 'last');
if isempty(k)
    error('edge50:noCrossing', 'No crossing of %g between %d and %d', level, iInside, iOutside);
end
x = idx(k) + step*(v(k) - level)/(v(k) - v(k+1));   % linear interpolation
end

function grid_out = grid_rotate(grid_in, theta)

grid_out = grid_in;
% rotate grid
grid_out.xx =  grid_in.xx * cos(theta) - grid_in.yy * sin(theta);
grid_out.yy =  grid_in.xx * sin(theta) + grid_in.yy * cos(theta);
end