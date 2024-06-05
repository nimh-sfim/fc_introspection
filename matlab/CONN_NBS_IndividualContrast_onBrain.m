close all
clear all

% Start Parallel Pool (for efficiency)
% ------------------------------------
addpath('/opt/matlab/conn')
addpath('/opt/matlab/spm12')
contrast = 'Surr-Neg-Self_gt_Image-Pos-Others';
pos_edges_color = [237, 125, 49]/255;
neg_edges_color = [68, 114, 196]/255;
if contrast == 'Image-Pos-Others_gt_Surr-Neg-Self'
    edg_color = pos_edges_color
end
if contrast == 'Surr-Neg-Self_gt_Image-Pos-Others'
    edg_color = [0.0, 0.0, 0.0];%neg_edges_color
end

% Load Connectivity Matrix
% ------------------------
disp('++ Load Connectivity Matrix')
tic
fc_file  = ['../resources/conn/NBS_SbjAware_NBS_3p1_',contrast,'.txt']

fc = load(fc_file);
N_conns = sum(sum(fc))/2;
fprintf('++ Number of connections = %d\n', N_conns)
toc
% Load ROI Labsls
% ---------------
fid        = fopen('../resources/conn/roi_labels.txt');
roi_labels = textscan(fid,'%s');
fclose(fid);
roi_labels = roi_labels{1,1};
roi_labels = roi_labels';

% Load ROI Labsls
% ---------------
roi_coords_arr=load('../resources/conn/roi_coords.txt');
roi_coords = {};
for i =1:380
    roi_coords{i} = roi_coords_arr(i,:);
end
global CONN_gui; CONN_gui.usehighres=true;

% Prepare input for plotting function
% -----------------------------------
rois.sph_xyz = roi_coords_arr; % Add Coordinates to input data structure
rois.sph_r = normalize(squeeze(transpose(sum(abs(fc)))), 'range',[.1,5]); % Add Radius based on degree
rois.sph_c = load('../resources/conn/roi_colors.txt');
disp('++ Opening the 3D Window')
tic
conn_display = conn_mesh_display('','','',rois,fc)
toc
disp('++ Set connections color')
tic
conn_display('con_color',edg_color)
toc
disp('Set ROI Transparency')
tic
conn_display('roi_transparency',0.7)
toc
disp('Set Brain Transparency')
tic
conn_display('brain_transparency',0.1)
toc
disp('Unset Subcorical surface')
tic
conn_display('sub_transparency',0)
toc
conn_display('con_bundling',1)
conn_display('con_width',0.1)
conn_display('con_transparency',.1)
conn_display('view',[0,0,1])
conn_display('background',[1,1,1])
conn_display('material',[]) % Equivalent to selecting Flat

%% ONLY RESULTS FOR HIGHER DEGREE NODE
% Not included as extra figure
close all
clear all

% Start Parallel Pool (for efficiency)
% ------------------------------------
addpath('/opt/matlab/conn')
addpath('/opt/matlab/spm12')
contrast = 'Surr-Neg-Self_gt_Image-Pos-Others';
pos_edges_color = [237, 125, 49]/255;
neg_edges_color = [68, 114, 196]/255;
if contrast == 'Image-Pos-Others_gt_Surr-Neg-Self'
    edg_color = pos_edges_color
end
if contrast == 'Surr-Neg-Self_gt_Image-Pos-Others'
    edg_color = [0.0, 0.0, 0.0];%neg_edges_color
end

% Load Connectivity Matrix
% ------------------------
disp('++ Load Connectivity Matrix')
tic
fc_file  = ['../resources/conn/NBS_SbjAware_NBS_3p1_',contrast,'.txt'];

fc = load(fc_file);
N_conns = sum(sum(fc))/2;
fprintf('++ Number of connections = %d\n', N_conns)
toc

% Detect higher degree node
sum_fc = sum(fc);
[d,r] = max(sum_fc);
fc2 = zeros(size(fc));
fc2(:,r) = fc(:,r);
fc2(r,:) = fc(r,:);
fc = fc2;

non_zero_rois = find(~all(fc==0,2));

fc = fc(non_zero_rois,non_zero_rois);
% Load ROI Labsls
% ---------------
fid        = fopen('../resources/conn/roi_labels.txt');
roi_labels = textscan(fid,'%s');
fclose(fid);
roi_labels = roi_labels{1,1};
roi_labels = roi_labels';

roi_labels = roi_labels(non_zero_rois);

% Load ROI Labsls
% ---------------
roi_coords_arr=load('../resources/conn/roi_coords.txt');
roi_coords = {};
for i =1:380
    roi_coords{i} = roi_coords_arr(i,:);
end

roi_coords = roi_coords(non_zero_rois);
roi_coords_arr = roi_coords_arr(non_zero_rois,:);
%
global CONN_gui; CONN_gui.usehighres=true;

% Prepare input for plotting function
% -----------------------------------
rois.sph_xyz = roi_coords_arr; % Add Coordinates to input data structure
rois.sph_r = normalize(squeeze(transpose(sum(abs(fc)))), 'range',[3,6]); % Add Radius based on degree
rois_colors = load('../resources/conn/roi_colors.txt');
rois.sph_c = rois_colors(non_zero_rois,:);

disp('++ Opening the 3D Window')
tic
conn_display = conn_mesh_display('','','',rois,fc)
toc
disp('++ Set connections color')
tic
conn_display('con_color',edg_color)
toc
disp('Set ROI Transparency')
tic
conn_display('roi_transparency',0.7)
toc
disp('Set Brain Transparency')
tic
conn_display('brain_transparency',0.1)
toc
disp('Unset Subcorical surface')
tic
conn_display('sub_transparency',0)
toc
conn_display('con_bundling',1)
conn_display('con_width',0.1)
conn_display('con_transparency',.1)
conn_display('view',[0,0,1])
conn_display('background',[1,1,1])
conn_display('material',[]) % Equivalent to selecting Flat
