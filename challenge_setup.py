from fastai.vision import *
from pathlib import Path
import pandas as pd

path = Path('/home/fabsta/projects/datascience/competitions/kaggle_human_protein_segmentation/input')
train = Path(path/'train')
test = Path(path/'test')
train_csv = Path(path/'train.csv')
#df = pd.read_csv(p/'train.csv')
sample = Path(path/'sample_submission.csv')
tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)

