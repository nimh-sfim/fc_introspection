%% Transform results so that they can be loaded in BrianNet
% In ther pre-print version we use CONN to visualize connectivity on glass brains
% This code remains here just in case we decide to revert back to BrainNet for plotting purposes
%  ===============================================================================================
clear all
RESOURCES_DIR="/data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/resources/nbs";
atlases=["Schaefer2018_400Parcels_7Networks_AAL2"];
designs=["SbjAware"];
cluster_solution = "CL02";
contrasts=["Image-Pos-Others_gt_Surr-Neg-Self","Surr-Neg-Self_gt_Image-Pos-Others"];
thresholds = ["NBS_3p1", "FDR_0p05"];
for atlas = atlases
    for threshold = thresholds
        for design = designs
            for contrast = contrasts
                work_path = fullfile(RESOURCES_DIR,atlas,"NBS_"+cluster_solution+"_Results",threshold,design);
                mat_file  = fullfile(work_path, "NBS_"+cluster_solution+"_"+contrast+".mat");
                txt_file  = fullfile(work_path,"NBS_"+cluster_solution+"_"+contrast+".txt");
                edge_file = fullfile(work_path,"NBS_"+cluster_solution+"_"+contrast+".edge");
                if exist(mat_file, 'file') == 2
                    % File exists.
                    data = load(mat_file);
                    num_networks = data.nbs.NBS.n;
                    switch num_networks
                        case 0
                            disp("++ INFO: ["+atlas+","+design+","+threshold+","+contrast+"] --> No Significant connections.")
                        case 1
                            data = full(cell2mat(data.nbs.NBS.con_mat));
                            data = data + data.';
                            writematrix(data,txt_file,"Delimiter"," ");
                            movefile(txt_file,edge_file);
                            disp("++ INFO: ["+atlas+","+design+","+threshold+","+contrast+"] --> Number of Significant Connections is " + (sum(sum(data))/2) + " originated in " + sum(sum(data)>0) + " nodes.")
                        otherwise
                            disp("++ WARNING: Selecting only the first network, but there are more.")
                            data = full(cell2mat(data.nbs.NBS.con_mat));
                            data = data + data.';
                            writematrix(data,txt_file,"Delimiter"," ");
                            movefile(txt_file,edge_file);
                            disp("++ INFO: ["+atlas+","+design+","+threshold+","+contrast+"] --> Number of Significant Connections is " + (sum(sum(data))/2) + " originated in " + sum(sum(data)>0) + " nodes.")
                    end
                else
                    % File does not exist.
                    disp("++ INFO: ["+atlas+","+design+","+threshold+","+contrast+"] --> No file available.")
                end
            end
        end
    end
end
