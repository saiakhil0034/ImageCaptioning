# image-captioning
Using RNN for generating captions for images


Run get_datasets-Jupyter notebook
this will generate: ./data/train('val/test')_ids.txt, 
and annotations folder

# Download the dataset
Run all the cells in get_datasets.ipynb to load the data in the root directory

# Training
With configurations as defined in config.py, run python train_with_checkpoints.py

# Generating Captions
Load the best model using load_bestmodel function in nntools.py and run evaluate in test mode for generating captions
