%% Transform results so that they can be loaded in BrianNet
% In ther pre-print version we use CONN to visualize connectivity on glass brains
% This code remains here just in case we decide to revert back to BrainNet for plotting purposes
%  ===============================================================================================
clear all
RESOURCES_DIR="/data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/resources/nbs";
atlases=["Schaefer2018_400Parcels_7Networks_AAL2"];
contrasts=["SetA_gt_SetB","SetB_gt_setA"];
for atlas = atlases
            for contrast = contrasts
                work_path = fullfile(RESOURCES_DIR,atlas,"NBS_Results");
                mat_file  = fullfile(work_path, "NBS_"+contrast+".mat");
                txt_file  = fullfile(work_path,"NBS_"+contrast+".txt");
                edge_file = fullfile(work_path,"NBS_"+contrast+".edge");
                if exist(mat_file, 'file') == 2
                    % File exists.
                    data = load(mat_file);
                    num_networks = data.nbs.NBS.n;
                    switch num_networks
                        case 0
                            disp("++ INFO: ["+atlas+","+contrast+"] --> No Significant connections.")
                        case 1
                            data = full(cell2mat(data.nbs.NBS.con_mat));
                            data = data + data.';
                            writematrix(data,txt_file,"Delimiter"," ");
                            movefile(txt_file,edge_file);
                            disp("++ INFO: ["+atlas+","+contrast+"] --> Number of Significant Connections is " + (sum(sum(data))/2) + " originated in " + sum(sum(data)>0) + " nodes.")
                        otherwise
                            disp("++ WARNING: Selecting only the first network, but there are more.")
                            data = full(cell2mat(data.nbs.NBS.con_mat));
                            data = data + data.';
                            writematrix(data,txt_file,"Delimiter"," ");
                            movefile(txt_file,edge_file);
                            disp("++ INFO: ["+atlas+","+contrast+"] --> Number of Significant Connections is " + (sum(sum(data))/2) + " originated in " + sum(sum(data)>0) + " nodes.")
                    end
                else
                    % File does not exist.
                    disp("++ INFO: ["+atlas+","+contrast+"] --> No file available.")
                end
            end
end
