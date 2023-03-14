import os
import numpy as np
import time
import argparse
import pickle
import glob
from datetime import datetime



parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, help="matrix file with key 'P' and 'data_mask' ", required=True)
args = parser.parse_args()

datafile = args.f



dimension_list = [2, 3, 4, 5, 6, 7, 8]
sparsity_list = [0, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]

J = len(glob.glob("./masks/*-0-*.npz"))
total_mask = len(glob.glob("./masks/*.npz"))
repeat = int(total_mask/J)
now = datetime.now()
dt_string = now.strftime("%d%m%Y-%H%M%S")
file = open("./swarm_file"+dt_string+".swarm","w")

for dimension in dimension_list:
    for sparsity in sparsity_list:
        for t in range(repeat):
            for j in range(J):
                file.write("cd ./; python model_cv.py -f {} -d {} -s {} -r {} -j {}\n"
                           .format(datafile, dimension, sparsity, t, j))

file.close()