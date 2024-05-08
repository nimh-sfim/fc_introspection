from nimare.dataset import Dataset 
from nimare import annotate
from nimare.extract import download_abstracts
from utils.basics import PRJ_DIR
import os.path as osp
import pickle

n_topics = 50 
vocab = 'LDA'+str(n_topics)
RESOURCE_NIMARE_DIR  = osp.join(PRJ_DIR,'nimare')
VOCAB_DIR            = osp.join(RESOURCE_NIMARE_DIR,vocab)

# Path to Neurosynth Dataset created in prior notebook
ns_dset_path  = osp.join(VOCAB_DIR,f'neurosynth_dataset_{vocab}.pkl.gz')
lda_dset_path = osp.join(VOCAB_DIR,f'lda_model_{vocab}.pkl.gz')

# Print information about this execution
print('++ INFO: Number of topics:        %d topics' % n_topics)
print('++ INFO: Neurosynth Dataset path: %s' % ns_dset_path)
print('++ INFO: LDA Model output file  : %s' % lda_dset_path)
# Load dataset into memory
print(f'++ INFO [{vocab}]: Load NeuroSynth Dataset into memory....')
dset=Dataset.load(ns_dset_path)

# Download abstracts for the papers in the NS dataset
print(f'++ INFO [{vocab}]: Downloading all abstracts...')
abstracts = download_abstracts(list(dset.ids),'javiergc@mail.nih.gov')

# Add a couple extra columns so the abstract dataframe looks like the one in the Laird Example (NiMare website)
abstracts['id'] = abstracts['study_id']+'-1'
abstracts['constrast_id'] = 1

# Append abstracts to dataset object
dset.texts = abstracts

# Create LDA Model
print(f'++ INFO [{vocab}]: Create LDA model...')
model = annotate.lda.LDAModel(n_topics=n_topics, max_iter=1000, text_column="abstract")

# Fit LDA Model
print(f'++ INFO [{vocab}]: Fit LDA model...')
new_dset = model.fit(dset)

# Save Results to disk
print(f'++ INFO [{vocab}]: Save results to disk...')
outputs  = {'model':model,'new_dset':new_dset}
with open(lda_dset_path,'wb') as f:
    pickle.dump(outputs,f)

print(f'++ INFO [{vocab}] Program finished sucessfully...')

