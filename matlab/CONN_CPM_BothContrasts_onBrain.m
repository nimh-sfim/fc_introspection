close all
clear all
what_to_show = 'neg'; % Options: 'pos','neg','both'
pred_target = 'Images';

% Start Parallel Pool (for efficiency)
% ------------------------------------
addpath('/opt/matlab/conn')
addpath('/opt/matlab/spm12')

% Colors for NBS results
pos_edges_color = [237, 125, 49]/255;
neg_edges_color = [68, 114, 196]/255;

% Colors for CPM models
pos_edges_color = [100, 9, 0]/255;
neg_edges_color = [9, 0, 100]/255;

% Load Connectivity Matrix
% ------------------------
disp('++ Load Connectivity Matrix')
tic
fc_path     = ['../resources/conn/CPM_',pred_target,'_matrix.txt'];
fc          = load(fc_path);
N_pos_conns = sum(sum(fc==1))/2;
N_neg_conns = sum(sum(fc==-1))/2;
N_conns = sum(sum(abs(fc)))/2;
fprintf('++ Number of connections = %d\n', N_conns)
fprintf('++ Number of pos connections = %d\n', N_pos_conns)
fprintf('++ Number of neg connections = %d\n', N_neg_conns)

toc

% Create Color Matrix
% -------------------
if strcmp(what_to_show,'both')
    disp('++ INFO: Preparing to show both models')
    edg_color = vertcat(repmat(neg_edges_color,N_neg_conns,1), repmat(pos_edges_color,N_pos_conns,1));
end
if strcmp(what_to_show,'pos')
    edg_color = pos_edges_color
    fc = (fc==1)
end
if strcmp(what_to_show,'neg')
    edg_color = neg_edges_color
    fc = (fc==-1)
end
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
