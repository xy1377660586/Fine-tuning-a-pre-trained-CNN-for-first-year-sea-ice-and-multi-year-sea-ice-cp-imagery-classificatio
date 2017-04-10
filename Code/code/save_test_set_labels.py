#coding=utf-8
# at first, you need to adjust the fc8 and fc6 and the output classes in the last, cause in the train_val.prototxt file you retained these layers in fact, therefore, the deploy.prototxt should consistant with the the train_val.prototxt file. remember that, it is the most important concepts.

import numpy as np
import scipy.io as sio
import sys,os

#
caffe_root = '/home/lein/caffe/' 
sys.path.insert(0, caffe_root + 'python')
import caffe
os.chdir(caffe_root)

net_file=caffe_root + 'myfinetunning/data/deploy.prototxt'
caffe_model=caffe_root + 'myfinetunning/snapshot111/_iter_100.caffemodel'
mean_file=caffe_root + 'myfinetunning/data/mean.npy'

net = caffe.Net(net_file,caffe_model,caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
transformer.set_raw_scale('data', 255) 
transformer.set_channel_swap('data', (2,1,0))

imagenet_labels_filename = caffe_root + 'myfinetunning/data/labels.txt'
labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

# get all the images in the 'all' folder.
def getAllImages(folder):
    assert os.path.exists(folder)
    assert os.path.isdir(folder)
    imageList = os.listdir(folder)

    return imageList
#####################################
imagelist1=getAllImages('/home/lein/Documents/remote_sensing/project/ice_cp_data/CP_9x9_332/first_year_patches')
firpatch_size=len(imagelist1)
pid=0
returnmat1=np.zeros(firpatch_size)
for i in range(0, firpatch_size):

    image_root1=imagelist1[pid]
    im=caffe.io.load_image('/home/lein/Documents/remote_sensing/project/ice_cp_data/CP_9x9_332/first_year_patches/'+image_root1)

    net.blobs['data'].data[...] = transformer.preprocess('data',im)
    out = net.forward()
    prob= net.blobs['prob'].data[0].flatten()#取出最后一层（Softmax）属于某个类别的概率值

    top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
    returnmat1[i]=top_k[0]
    pid=pid+1
    print pid

sio.savemat('/home/lein/caffe/myfinetunning/data/firpre_labels',{'data':returnmat1})

#print top_k[0]
#print prob
#####################################

imagelist2=getAllImages('/home/lein/Documents/remote_sensing/project/ice_cp_data/CP_9x9_332/multi_year_patches')

mulpatch_size=len(imagelist2)
pi=0
returnmat2=np.zeros(mulpatch_size)
for j in range(0,mulpatch_size):
    image_root2=imagelist2[pi]
    im2=caffe.io.load_image('/home/lein/Documents/remote_sensing/project/ice_cp_data/CP_9x9_332/multi_year_patches/'+image_root2)

    net.blobs['data'].data[...] = transformer.preprocess('data',im2)
    out = net.forward()
    prob= net.blobs['prob'].data[0].flatten()#取出最后一层（Softmax）属于某个类别的概率值

    top_k2 = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
    returnmat2[j]=top_k2[0]
    pi=pi+1
    print pi

sio.savemat('/home/lein/caffe/myfinetunning/data/mulpre_labels',{'data':returnmat2})



