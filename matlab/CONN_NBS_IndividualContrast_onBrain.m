close all
clear all

% Start Parallel Pool (for efficiency)
% ------------------------------------
%addpath('/opt/matlab/conn')
%addpath('/opt/matlab/spm12')
contrast = 'SetA_gt_SetB';
pos_edges_color = [237, 125, 49]/255;
neg_edges_color = [68, 114, 196]/255;
if contrast == 'SetA_gt_SetB'
    edg_color = [0.0, 0.0, 0.0];
end
if contrast == 'SetB_gt_SetA'
    edg_color = [0.0, 0.0, 0.0];%neg_edges_color
end

% Load Connectivity Matrix
% ------------------------
disp('++ Load Connectivity Matrix')
tic
fc_file  = ['../resources/conn/NBS_T3p1_s0.05_',contrast,'.txt']

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
