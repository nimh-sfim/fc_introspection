from nimare.dataset import Dataset 
from nimare import annotate
from nimare.extract import download_abstracts
import pickle

n_topics = 400 
vocab = 'LDA'+str(n_topics)
# Path to Neurosynth Dataset created in prior notebook
ns_dset_path = f'/data/SFIMJGC_Introspec/prj2021_dyneusr/Resources_NiMare/{vocab}/neurosynth_dataset.pkl.gz'

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
out_path = f'/data/SFIMJGC_Introspec/prj2021_dyneusr/Resources_NiMare/{vocab}/lda_model.pkl.gz'
with open(out_path,'wb') as f:
    pickle.dump(outputs,f)

print('++ INFO [{vocab}] Program finished sucessfully...')

