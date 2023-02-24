import os
import numpy as np
import time
import argparse
import pickle
import glob
from datetime import datetime
import getpass
import os.path as osp
from basics import PRJ_DIR

# Parse Input arguments
# =====================
parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, help="matrix file with key 'P' and 'data_mask' ", required=True)
args = parser.parse_args()

datafile = args.f

# Configuration of ranges of hyper-parameters to explore
# ======================================================
dimension_list = [2, 3, 4, 5]
sparsity_list =  [0.00001, 0.001, 0.01, 0.1, 1] #[0.001, 0.003, 0.005, 0.007, 0.009, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0]
#sparsity_list =  [0.00001]

# Create/Obtain values necessary for drafting the swarm calls
# ===========================================================
username    = getpass.getuser()
print("++ INFO: username    = %s" % username)
logs_folder = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))
print("++ INFO: Logs folder = %s" % logs_folder)
J = len(glob.glob("{PRJ_DIR}/data/snycq/masks/*-0-*.npz".format(PRJ_DIR=PRJ_DIR)))
print('++ INFO: J           = %d' % J)
total_mask = len(glob.glob("{PRJ_DIR}/data/snycq/masks/*.npz".format(PRJ_DIR=PRJ_DIR)))
repeat = int(total_mask/J)
print('++ INFO: Repeat      = %d' % repeat)
now             = datetime.now()
dt_string       = now.strftime("%d%m%Y-%H%M%S")
swarm_file_path = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username),"SNYCQ_Model_cv.SWARM.sh")
#swarm_file_path = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username),"swarm_file."+dt_string+".swarm")
print("++ INFO: Swarm File  = %s" % swarm_file_path)


# Create the Swarm File
# =====================
file            = open(swarm_file_path,"w")
file.write('#swarm -f {swarm_file_path} -b 24 --logdir {logs_folder} --gb-per-process 24 --noht --partition quick --time 00:05:00 --gres=lscratch:10\n'.format(swarm_file_path=swarm_file_path,
                                                                                                                                                       logs_folder=logs_folder))
for dimension in dimension_list:
    for sparsity in sparsity_list:
        for t in range(repeat):
            for j in range(J):
                file.write("export VAR_D={d} VAR_S={s} VAR_R={r} VAR_J={j} VAR_FILE={datafile}; sh {PRJ_DIR}/code/fc_introspection/notebooks/utils/SNYCQ_model_cv.sh\n"
                           .format(datafile=datafile,PRJ_DIR=PRJ_DIR,
                                   d=dimension,
                                   s=sparsity,
                                   r=t,
                                   j=j))

file.close()
print('++ INFO: Swarm file saved in [%s]' % swarm_file_path)
print('++ INFO: To run the jobs, you can type the following:')
print('         swarm -f {swarm_file_path} -b 24 --logdir {logs_folder} --gb-per-process 24 --noht --partition quick --time 00:05:00 --gres=lscratch:10'.format(swarm_file_path=swarm_file_path,
                                                                                                                                                       logs_folder=logs_folder))
