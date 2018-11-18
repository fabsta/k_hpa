# Human protein atlas image classification

[competition link](https://www.kaggle.com/c/human-protein-atlas-image-classification)

Interesting kernels:
* [fastai starter](http://nbviewer.jupyter.org/github/fabsta/interesting_notebooks/blob/master/pretrained-resnet34-with-rgby-0-460-public-lb.ipynb)
* [fastai course multi-classification example](http://nbviewer.jupyter.org/github/fastai/fastai/blob/master/courses/dl1/lesson2-image_models.ipynb#Multi-label-classification)

useful links
* [data augmentation](https://becominghuman.ai/data-augmentation-using-fastai-aefa88ca03f1)


Workflow
- [ ]  Import libraries
- [ ]  Define data path,
- [ ]  Defining data loader
- [ ]  Define focal loss and accuracy
- [ ]  Define custom architecture
- [ ]  Get learner ready
- [ ]  Start training
- [ ]  Train head of model
- [ ]  Unfreeze all weights and train entire model
- [ ]  Test time augmentation
- [ ]  Validation, F1 score
- [ ] Submission


Things to check out:
gpu-stats
in python notebook

github todo lists

## Ideas grouped by topic

Data
- [ ] sample dataset
- [ ] Use more data: [A dataset of images and morphological profiles of 30 000 small-molecule treatments using the Cell Painting assay](https://academic.oup.com/gigascience/article/6/12/giw014/2865213)

Preprocessing
- [ ] Discarding yellow runs faster, doesn't change result. more discards?
- [ ] Merge two semantically similar channels

Training
- [ ] Progressively increase image size
- [ ] Stratified training data https://github.com/trent-b/iterative-stratification 
- [ ] Cross validation example [here](https://github.com/radekosmulski/tgs_salt_solution/blob/master/unet34_like_128.ipynb)
- [ ] Other data augmentation techniques (cropping images), what would be logical?
- [ ] Find optimal weight decay: [link](http://nbviewer.jupyter.org/github/MicPie/lung/blob/master/lung_inflammation_v4_ResNet34.ipynb)

Model
- [ ] 4-Channel model: swap first layer: [link](https://forums.fast.ai/t/lesson-3-in-class-discussion/7809/86?u=jpjamipark).  I added simple ConvBlock reducing from 4 to 3 channels before pretrained model. Works good.@ryches I freeze pretrained network, and set this Convlayer and last layers to trainable. https://forums.fast.ai/t/how-to-do-transfer-learning-with-different-inputs/28395/3
- [ ] Save best model
- [ ] Use other pre-trained model? https://github.com/Cadene/pretrained-models.pytorch 
- [ ] Fastai v1 starter pack: [kaggle](https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/71039), [github](https://github.com/wdhorton/protein-atlas-fastai), [notebook](http://nbviewer.jupyter.org/github/wdhorton/protein-atlas-fastai/blob/master/resnet50_basic.ipynb)
- [ ] other notebook: [lesson2-protein-human-protein-atlas-v1_256-resnet34.ipynb](http://nbviewer.jupyter.org/github/artste/fastai-samples/blob/master/kaggle/lesson2-protein-human-protein-atlas-v1_256-resnet34.ipynb)
- [ ] another fastai starter: [link](http://nbviewer.jupyter.org/github/ademyanchuk/protein_atlas_baseline/blob/master/protein-1.ipynb)
- [ ] learn = create_cnn(data, arch, metrics=[acc_02, f_score]).to_fp16()
- [ ] Replace average pooling layer with adaptive average layer
- [ ] Papers: [GapNet-PL paper](https://openreview.net/pdf?id=ryl5khRcKm), [Cell organelle classification with fully convolutional neural networks](https://pdfs.semanticscholar.org/8015/5ab5da4c739541a4d6b97c0189355ca7d476.pdf)

Score
- [ ] Check if f1-score is used
- [ ] Threshold selection for multi-label classification, [paper](https://www.csie.ntu.edu.tw/~cjlin/papers/threshold.pdf)

other competitions

image preprocessing

## Notes from forum


filters:
green filter for prediction, others for reference
merging images improves score	
Discarding yellow runs faster, doesnt change result. more discards?
Green is the protein itself. The other colors are other parts of the cell. While they are not required, they can provide useful information.
yellow means endoplasmatic reticulum?



Train/Test data split:
multilabel stratification python package
class imbalance
“huge” difference between validation and test score
Use cross-validation to get better understanding of predictions on diff validation sets

f1 metric:
order of ids is important!
macro f1 score
sklearn.metrics.f1_score with average="macro"
focal loss + soft F1 and focal loss - log(soft F1) for faster convergence

LB:
LB probing using all labels benchmark

Postprocessing:
Threshold selection for multi-label classification


Improvement ideas:
- split big image into smaller
- make a network that will give me bounding boxes of cells to process. Then from the large scale images I can get smaller images of cells to train a network on.

- ensembl ideas (nasnet)
- 
if there the image contain e.g. "one" object of type A, it is has a label A.
if it contains two, three, …. objects, it is has still a label A

you can always break the image into smaller image an ensemble back again. My CDiscount challenge solution gives you a clue on how to do it!
- Use features from different resolutions

> Written with [StackEdit](https://stackedit.io/).
