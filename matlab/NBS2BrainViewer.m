%% Transform results so that they can be loaded in BrianNet
%  ========================================================
clear all
RESOURCES_DIR="/data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/resources/mtl_snycq/no_confounds/nbs";
atlases=["Schaefer2018_200Parcels_7Networks","Schaefer2018_200Parcels_7Networks_AAL2"];
scenarios=["CL02_0p005", "CL02_0p001"];
contrasts=["F1gtF2","F2gtF1"];
for atlas = atlases
    for scenario = scenarios
        for contrast = contrasts
            input_path   = RESOURCES_DIR + "/"+atlas+"/NBS_"+scenario+"_Results/NBS_"+scenario+"_"+contrast+".mat";
            output_path  = RESOURCES_DIR + "/"+atlas+"/NBS_"+scenario+"_Results/NBS_"+scenario+"_"+contrast+".txt";
            output_path2 = RESOURCES_DIR + "/"+atlas+"/NBS_"+scenario+"_Results/NBS_"+scenario+"_"+contrast+".edge";
            if exist(input_path, 'file') == 2
                % File exists.
                data = load(input_path);
                data = full(cell2mat(data.nbs.NBS.con_mat));
                data = data + data.';
                writematrix(data,output_path,"Delimiter"," ");
                movefile(output_path,output_path2);
                disp("++ INFO: ["+atlas+","+scenario+","+contrast+"] --> Number of Significant Connections is " + (sum(sum(data))/2) + " originated in " + sum(sum(data)>0) + " nodes.")
            else
                % File does not exist.
                disp("Data does not exists for ["+atlas+","+scenario+","+contrast+"]");
            end
        end
    end
end
