import os
import numpy as np
import pandas as pd
import glob
import pickle
import seaborn as sns
import time


from matplotlib import pyplot
from matplotlib.pyplot import *



# Load parsing data

result_list = glob.glob('./results/*.npy')
output_full = []
for (z, f) in enumerate(result_list):
    output = np.load(f)
    output_full.append(output)
    
output_full = np.stack(output_full)
    
mini_df = pd.DataFrame(np.vstack(output_full), columns=['Dimension','sparsity',
                                                    'train_error','valid_error'])

betalist = np.unique(mini_df['sparsity'].values)

fig, ax = pyplot.subplots(figsize=(12,4))
for b in betalist:
    tempdf = mini_df.loc[(mini_df['sparsity'] == b)]
    sns.lineplot(x="Dimension", y="valid_error",style="sparsity",data=tempdf,
                 ax=ax, label=str(b), legend=False, markers=True)
ax.set_ylabel('Frobenius norm (log-scale)')
ax.set_yscale('log')
ax.legend(prop={'size': 11}, loc='center', bbox_to_anchor=(0.5, 1), ncol=6, fancybox=True, shadow=True)
ax.set_title('Reconstruction Error', y=1.08)


fig.savefig("./hyper-parameters.pdf", bbox_inches='tight')