from fastai import *
from fastai.vision import *
#from fastai import *
import numpy as np
from PIL import *

def display_imgs(x):
    columns = 4
    bs = x.shape[0]
    rows = min((bs+3)//4,4)
    fig=plt.figure(figsize=(columns*4, rows*4))
    for i in range(rows):
        for j in range(columns):
            idx = i+j*columns
            fig.add_subplot(rows, columns, idx+1)
            plt.axis('off')
            plt.imshow((x[idx,:,:,:3]*255).astype(np.int))
    plt.show()

#def open_image4d(path:PathOrStr)->Image:
def open_image4d(path):
    '''open RGBA image from 4 different 1-channel files.
    return: numpy array [4, sz, sz]'''
    path=str(path)
    flags = cv2.IMREAD_GRAYSCALE
    red = cv2.imread(path+ '_red.png', flags)
    blue = cv2.imread(path+ '_blue.png', flags)
    green = cv2.imread(path+ '_green.png', flags)
    yellow = cv2.imread(path+ '_yellow.png', flags)

    im = np.stack(([red, green, blue, yellow]))

    return Image(Tensor(im/255).float())


def open_image3d(path):
    '''open RGBA image from 4 different 1-channel files.
    return: numpy array [4, sz, sz]'''
    path=str(path)
    flags = cv2.IMREAD_GRAYSCALE
    red = cv2.imread(path+ '_red.png', flags)
    blue = cv2.imread(path+ '_blue.png', flags)
    green = cv2.imread(path+ '_green.png', flags)
    #yellow = cv2.imread(path+ '_yellow.png', flags)

    im = np.stack(([red, green, blue]))
    
    return Image(Tensor(im/255).float())




def openMultiChannelImage(path, id):
    colors = ['red','green','blue','yellow']
    mat = None
    nChannels = len(colors)
    for i,color in enumerate(colors):
        curr_path = os.path.join(path, str(id)+'_'+color+'.png')
        print('Loading: ', curr_path)
        img = PIL.Image.open(curr_path)
        chan = pil2tensor(img).float().div_(255)
        if(mat is None):
            mat = torch.zeros((nChannels,chan.shape[1],chan.shape[2]))
        mat[i,:,:]=chan
#    return Image(mat)
    return mat

import cv2
def open_rgby(path,id,size=512): #a function that reads RGBY image
    colors = ['red','green','blue','yellow']
    flags = cv2.IMREAD_GRAYSCALE
    img = [cv2.imread(os.path.join(path, id+'_'+color+'.png'), flags)
           for color in colors]
    for i,im in enumerate(colors):
        if img[i] is None:
            img[i] = np.zeros( (512,512), dtype=np.uint8)
        else:
            img[i] = img[i].astype(np.float32)/255
    
    return np.stack(img, axis=-1)


### https://github.com/wdhorton/protein-atlas-fastai/blob/master/dataset.py
# adapted from https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
def open_4_channel(fname):
    fname = str(fname)
    # strip extension before adding color
    if fname.endswith('.png'):
        fname = fname[:-4]
    colors = ['red','green','blue','yellow']
    flags = cv2.IMREAD_GRAYSCALE
    img = [cv2.imread(fname+'_'+color+'.png', flags).astype(np.float32)/255
           for color in colors]
    
    x = np.stack(img, axis=-1)
    return Image(pil2tensor(x, np.float32).float())

class ImageMulti4Channel(ImageMultiDataset):
    def __init__(self, fns, labels, classes=None, **kwargs):
        super().__init__(fns, labels, classes, **kwargs)
        self.image_opener = open_4_channel
