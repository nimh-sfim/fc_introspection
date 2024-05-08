import os
from glob import glob
print("importing nilearn masking")
from nilearn import masking
print("successfully imported nilearn masking")
from nimare import dataset, meta
from nimare.extract import fetch_neurosynth
from nimare.io import convert_neurosynth_to_dataset
from nimare.stats import pearson
import pandas as pd
import numpy as np
import os.path as osp
import shutil
import argparse
from tqdm import tqdm

PRJDIR = "/data/SFIMJGC_Introspec/2023_fc_introspection"
#PRJDIR = "/data/SFIMJGC_Introspec/prj2021_dyneusr/Resources_NiMare/"

def run(args):
  
  vocab = args.vocab
  ml    = args.ml
  do_full_setup = args.full_setup
  verbose = args.verbose
  if args.topic is not None:
   TOPICS = [args.topic.replace('-',' ')]
  else:
   TOPICS = None
  
  RESOURCE_NIMARE_DIR  = osp.join(PRJDIR,'nimare')
  VOCAB_DIR            = osp.join(RESOURCE_NIMARE_DIR,vocab)
  METAMAPS_ORIG_DIR    = osp.join(VOCAB_DIR,"meta-analyses-orig")  # where to save meta-analysis maps

  #RESOURCE_DIR  = osp.join(PRJDIR,vocab)
  #NIMARE_PATH   = osp.join(RESOURCE_DIR,'nimare')
  #METAMAPS_DIR  = os.path.join(RESOURCE_DIR,"meta-analyses-orig")  # where to save meta-analysis maps
  
  # Log Execution Start Time
  print ("++ INFO: Vocabulary   = %s" % vocab)
  print ("++ INFO: Memory Limit = %s" % ml)
  if do_full_setup:
      print('++ INFO: Full Setup Requested (this will take hours and tons of memory. Not recommended as part of batch job.)')
  if TOPICS is not None:
      print('++ INFO: Specific Topic Requested [%s]' %args.topic)
    
  # Create Empty Output Folders
  # ===========================
  print('++ WARNING: TEMPORARY VOCAB LOCATION. PLEASE CHANGE TO DEFINITIVE!!!!!!')
  print('++ WARNING: TEMPORARY VOCAB LOCATION. PLEASE CHANGE TO DEFINITIVE!!!!!!')
  print('++ WARNING: TEMPORARY VOCAB LOCATION. PLEASE CHANGE TO DEFINITIVE!!!!!!')
  print('++ WARNING: TEMPORARY VOCAB LOCATION. PLEASE CHANGE TO DEFINITIVE!!!!!!')
  ns_dset_path = os.path.join(VOCAB_DIR, f"neurosynth_dataset_{vocab}_EXTRA.pkl.gz")
  print(ns_dset_path) 
  # ===============================================================================
  # FULL SETUP: Not necessary, as this should happen on the notebook interactively
  # ===============================================================================
  if do_full_setup:
      print("++ INFO: Setting up all necessary folders")
      for folder_path in [RESOURCE_NIMARE_DIR, VOCAB_DIR, METAMAPS_ORIG_DIR] :
      #for folder_path in [RESOURCE_DIR, NIMARE_PATH, METAMAPS_DIR]:
          if osp.exists(folder_path):
             print(" + WARNING: Removing folder [%s]" % folder_path)
             shutil.rmtree(folder_path)
          print(" + INFO: Generating/Regenerating output folder [%s]" % folder_path)
          os.mkdir(folder_path)
          
      # Download Neurosynth Data
      # ========================
      print("++ INFO: Fetching neurosynth dataset for this vocabulary...")
      #dset_file    = os.path.join(RESOURCE_DIR, "neurosynth_dataset.pkl.gz")
      #dset_ma_file = os.path.join(RESOURCE_DIR, "neurosynth_dataset_with_ma.pkl.gz")
    
      if not os.path.isfile(ns_dset_path):
        files = fetch_neurosynth(
            data_dir=VOCAB_DIR, 
            version="7",
            overwrite=False,
            source="abstract",
            vocab=vocab,
        )
        neurosynth_db = files[0]
    
        neurosynth_dset = convert_neurosynth_to_dataset(
            coordinates_file=neurosynth_db["coordinates"],
            metadata_file=neurosynth_db["metadata"],
            annotations_files=neurosynth_db["features"],
        )
        # Save the dataset as a pickle file to the Resources directory
        neurosynth_dset.save(ns_dset_path)
      else:
        print (" + Saving dataset to %s" % ns_dset_path)
        neurosynth_dset = dataset.Dataset.load(ns_dset_path)
       
      print(" + Neurosynth dataset: %s" % neurosynth_dset)
       
      # Update Neurosynth dataset path (for meta-analysis intermediate outputs)
      print("++ INFO: Updating Neurosynth Object path to [%s]" % NIMARE_PATH)
      neurosynth_dset.update_path(NIMARE_PATH)
      
      # Initialize the Estimator
      print("++ INFO: Initializing estimator")
      # You could use `memory_limit` here if you want, but that will slow things down.
      meta_estimator = meta.cbma.mkda.MKDAChi2(memory_limit=ml)
      
      # Pre-generate MA maps to speed the meta-analyses up. This step may take some time.
      # This step will create tons of temporary files in NIMARE_PATH
      # Independently of memory_limit, it looks like this steps makes mem usage to go up to approx 100GB
      # This software runs with NiMare 0.0.9... it breaks with 0.0.13 (current as of March 2023)
      print("++ INFO: Pre-generate MA maps to speed the meta-analyses up")
      kernel_transformer = meta_estimator.kernel_transformer
      neurosynth_dset = kernel_transformer.transform(neurosynth_dset, return_type="dataset")
    
      # Save neurosynth object updated with MA estimations
      print("++ INFO: Saving updated NeuroSynth object [%s]" % dset_ma_file)
      neurosynth_dset.save(dset_ma_file)
  # ==============================================================================
  # ALETERNATIVE TO FULL SETUP
  # ==============================================================================
  else:
      #dset_file    = os.path.join(RESOURCE_DIR, "neurosynth_dataset.pkl.gz")
      neurosynth_dset = dataset.Dataset.load(ns_dset_path)
      meta_estimator = meta.cbma.mkda.MKDAChi2()
  # Get features
  if TOPICS is None:
      print ("++ INFO: Get Dictionary Labels")
      TOPICS = neurosynth_dset.get_labels()
  for label in tqdm(TOPICS):
    print(" +      Processing {label}".format(label=label))
    # Use a threshold of 0.05 for topics, even though the default threshold is 0.001.
    label_positive_ids = neurosynth_dset.get_studies_by_label(label, 0.05)
    label_negative_ids = list(set(neurosynth_dset.ids) - set(label_positive_ids))

    # Require some minimum number of studies in each sample
    if (len(label_positive_ids) == 0) or (len(label_negative_ids) == 0):
        print("+        Skipping {label}".format(label=label))
        continue

    label_positive_dset = neurosynth_dset.slice(label_positive_ids)
    label_negative_dset = neurosynth_dset.slice(label_negative_ids)
    meta_result = meta_estimator.fit(label_positive_dset, label_negative_dset)
    # Save the maps to an output directory
    print(" +         Saving maps for label [%s]" % str(label))
    meta_result.save_maps(output_dir=METAMAPS_ORIG_DIR, prefix=label.replace(' ','-'))

# =======================================================================
# =====================  MAIN FUNCTION  =================================
# =======================================================================
def main():
  # Parse input arguments
  parser=argparse.ArgumentParser(description="Compute embeddings for a given run using a specific metric, k, time smooth strategy and tree type")
  parser.add_argument("-V",       help="Topic Vocabulary" ,dest="vocab",  type=str, required=True, choices=['LDA50','LDA400'])
  parser.add_argument("-ml",          help="Memory Limit"     ,dest="ml",     type=str, required=False, choices=['1gb','5gb','10gb'], default=None)
  parser.add_argument("-t", help='Topic', dest='topic', type=str, required=False, default=None)
  parser.add_argument("-s", help='Do Setup from the begining. Otherwise, assume you can jump right away to creating topic maps', action='store_true', dest='full_setup',required=False)
  parser.add_argument('-v','--verbose', action='store_true', help="Show additional information on screen", dest='verbose', required=False)
  parser.set_defaults(verbose=False)
  parser.set_defaults(full_setup=False)
  parser.set_defaults(func=run)
  args=parser.parse_args()
  args.func(args)

if __name__ == "__main__":
    main()