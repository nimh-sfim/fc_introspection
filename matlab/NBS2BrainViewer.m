%% Transform results so that they can be loaded in BrianNet
%  ========================================================
clear all
RESOURCES_DIR="/data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/resources/nbs";
atlases=["Schaefer2018_400Parcels_7Networks_AAL2","Schaefer2018_200Parcels_7Networks_AAL2"];
scenarios=["All_Scans"];
cluster_solution = "CL02";
contrasts=["Image-Pos-Others_gt_Surr-Neg-Self","Surr-Neg-Self_gt_Image-Pos-Others"];
thresholds = ["NBS_3p1","NBS_3p5","FDR_0p05","NBS_3p1_augmented","NBS_3p5_augmented","NBS_2p7_augmented"];
for atlas = atlases
    for scenario = scenarios
        for threshold = thresholds
            for contrast = contrasts
                work_path = fullfile(RESOURCES_DIR,atlas,scenario,"NBS_"+cluster_solution+"_Results",threshold);
                mat_file  = fullfile(work_path, "NBS_"+cluster_solution+"_"+contrast+".mat");
                txt_file  = fullfile(work_path,"NBS_"+cluster_solution+"_"+contrast+".txt");
                edge_file = fullfile(work_path,"NBS_"+cluster_solution+"_"+contrast+".edge");
                if exist(mat_file, 'file') == 2
                    % File exists.
                    data = load(mat_file);
                    num_networks = data.nbs.NBS.n;
                    if num_networks == 1
                        data = full(cell2mat(data.nbs.NBS.con_mat));
                    else
                        data = full(cell2mat(data.nbs.NBS.con_mat(1)));
                        disp("++ WARNING: Selecting only the first network, but there are more.")
                    end
                    data = data + data.';
                    writematrix(data,txt_file,"Delimiter"," ");
                    movefile(txt_file,edge_file);
                    disp("++ INFO: ["+atlas+","+scenario+","+threshold+","+contrast+"] --> Number of Significant Connections is " + (sum(sum(data))/2) + " originated in " + sum(sum(data)>0) + " nodes.")
                else
                    % File does not exist.
                    disp("Data does not exists for ["+atlas+","+scenario+","+threshold+","+contrast+"]");
                end
            end
        end
    end
end
