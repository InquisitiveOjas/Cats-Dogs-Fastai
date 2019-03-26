#Importing all the external libraries
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

path = "data/dogscats/"

sz = 224

#To reset the precomputed activations
!rm -rf {path}tmp

arch = resnet34
data = ImageClassifierData.from_paths(path , tfms=tfms_from_model(arch,sz))
learn = ConvLearner.pretrained(arch , data , precompute = True)
learn.fit(0.01 , 3) # 3 epochs

#labels for validation data
da.val_yal

#from here we know that dogs are labelled as "1" and cats are as labelled as "0"
data.classes #['cats', 'dogs']

#This gives prediction for validation set. Predictions are in log scale
log_preds = learn.predict()
log_preds.shape #(2000,2)

log_preds[:10]

preds = np.argmax(log_preds , axis = 1) #from log probabilities to 0 or 1
probs = np.exp(log_preds[:..1]) #pr(dog)

def rand_by_mask(mask): 
  return np.random.choice(np.where(mask)[0] , 4 , replace = False)
  
def rand_by_correct (is_correct):
  return rand_by_mask((preds == data.val_y) == is_correct)
  
def plot_val_with_title (idxs , title):
    imgs = np.stack([data.val_ds[x][0] for x in idxs])
    title_probs = [probs[x] for x in idxs]
    print (title)
    return plots (data.val_ds.denorm(imgs) , rows = 1 , titles = title_probs)
    
def plots (ims , figsize = (12,6) , rows = 1 , titles = None):
    f = plt.figure(figsize = figsize)
    for i in range (len(ims)):
        sp = f.add_subplot(rows , len(ims)//rows , i+1)
        sp.axis('Off')
        if titles is not None : sp.set_title (titles[i] , fontsize = 16)
        plt.imshow(ims[i])

def load_img_id (ds , idx):
    return np.array(PIL.Image.open(PATH = ds.fnames[idx]))
    
def plot_val_with_title (idxs , title):
    imgs = [load_img_id(data.val_ds,x) for x in idxs]
    title_probs = [probs[x] for x in idxs]
    print (title)
    return plots (imgs , rows = 1 , titles = title_probs , figsize = (16,8))

# 1. A few correct labels at  random
plot_val_with_title(rand_by_correct(True) , "Correctly Classified")

# 2. A few incorrect labels at  random
plot_val_with_title(rand_by_correct(False) , "Incorrectly Classified")

def most_by_mask (mask , mult):
    idxs = np.where(mask)[0]
    return idxs[np.argsort(multi * probs[idxs])[:4]]
    
def most_by_correct (y , is_correct):


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        






























