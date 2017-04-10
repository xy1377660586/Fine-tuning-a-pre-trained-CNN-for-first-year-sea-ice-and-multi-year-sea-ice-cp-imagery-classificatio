import os, re, time, sys
import string
import numpy  as np
import scipy.io as sio

# get all the images in the 'all' folder.
def getAllImages(folder):
    assert os.path.exists(folder)
    assert os.path.isdir(folder)
    imageList = os.listdir(folder)

    return imageList
#####################################
imagelist1=getAllImages('/home/lein/Documents/remote_sensing/project/ice_cp_data/CP_9x9_332/multi_year_patches')
print imagelist1[0:10]
mulpatch_size=len(imagelist1)

index_mul=np.zeros(mulpatch_size)
jj=0
for i in range(0, mulpatch_size):
    temp=imagelist1[jj]
    loc=temp[24]
    if temp[25]!='.':
        loc=loc*1+temp[25]
    if temp[26]!='.' and temp[26]!='j':
        loc=loc*1+temp[26]
    if temp[27]!='.' and temp[27]!='j' and temp[27]!='p':
        loc=loc*1+temp[27]
    index_mul[jj]=loc
    jj=jj+1

    print loc
   
    
sio.savemat('/home/lein/caffe/myfinetunning/data/mul_index',{'data':index_mul})

print index_mul
