import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm
import scipy.misc
import imageio
from PIL import Image
import scipy.io
import os
import cv2
from skimage import measure
from skimage.filters import try_all_threshold
from tqdm import tqdm
# Make sure that caffe is on the python path:
#caffe_root = '../../../'
import sys
import os
#sys.path.insert(0, caffe_root + 'python')
import caffe
EPSILON = 1e-8

#remove the following two lines if testing with cpu
caffe.set_mode_gpu()
# choose which GPU you want to use
caffe.set_device(0)
caffe.SGDSolver.display = 0
# load net
net = caffe.Net('deploy.prototxt', 'dss_model_released.caffemodel', caffe.TEST)

#****************************need modify***************************************
#when reusing this script, please modify values of outRootPath and dataSetPath
#there will be a '/' followed by the path
#path to store the output image
outRootPath = '/data/data-hulishuang/test-images/output/'
#outRootPath = '/home/hls/test/'

#the root path of iamge dataset
dataSetPath = '/data/data-hulishuang/test-images/input/'
#dataSetPath = '/home/hls/generative_inpainting/examples/places2/'

#*******************************************************************************

S = 600

images = os.listdir(dataSetPath)
N_images = len(images)
#for i in range(2):
for i in tqdm(range(N_images), file=sys.stdout, leave=False, dynamic_ncols=True):    
    
    # load image
    img = Image.open(dataSetPath + images[i])
    #img = Image.open('/home/hls/caffe/data/MSRA-B/9_ss07043.jpg')
    
    #resize the image shape such that the largest side equals S
    width,height = img.size

    im_size_wh = np.array([width,height])
    ratio = float(S)/np.max(im_size_wh)
    new_size = tuple (np.round(im_size_wh * ratio).astype(np.int32) )
    im = img.resize(new_size,Image.ANTIALIAS)

    #Preprocessing
    im_tmp = np.array(im, dtype=np.uint8)
    im = np.array(im_tmp, dtype=np.float32)
    im = im[ :, :,::-1]
    #means for msra-b
    #im -= np.array((104.00698793,116.66876762,122.67891434))
    #means for oxford
    im -= np.array((103.93900299,  116.77899933,  123.68000031))
    im = im.transpose((2,0,1))



    # load gt
    #gt = Image.open('/home/hls/caffe/data/MSRA-B/9_ss07087.png')

    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *im.shape)
    net.blobs['data'].data[...] = im
    # run net and take argmax for prediction
    net.forward()
    out1 = net.blobs['sigmoid-dsn1'].data[0][0,:,:]
    out2 = net.blobs['sigmoid-dsn2'].data[0][0,:,:]
    out3 = net.blobs['sigmoid-dsn3'].data[0][0,:,:]
    out4 = net.blobs['sigmoid-dsn4'].data[0][0,:,:]
    out5 = net.blobs['sigmoid-dsn5'].data[0][0,:,:]
    out6 = net.blobs['sigmoid-dsn6'].data[0][0,:,:]
    fuse = net.blobs['sigmoid-fuse'].data[0][0,:,:]
    res = (out3 + out4 + out5 + fuse) / 4
    res = (res - np.min(res) + EPSILON) / (np.max(res) - np.min(res) + EPSILON)


    
    #threshold
    #fig,ax = try_all_threshold(res,figsize=(10,8),verbose=False)
    binary = res > 0.8
    
    labels = measure.label(binary, connectivity =2)
    regions = measure.regionprops(labels)
    max_area_index = -1
    max_area = -1
    count = -1
    for props in regions:
        count = count + 1
        if props.area > max_area:
            max_area = props.area
            max_area_index = count
    #find the centroid of largest region, then make a mask around the center
    half_length = 25
    x0,y0 = regions[max_area_index].centroid
    x0 = int (x0)
    y0 = int (y0)
    minr = x0 - half_length
    minc = y0 - half_length
    maxr = x0 + half_length
    maxc = y0 + half_length
    #minr,minc,maxr,maxc = regions[max_area_index].bbox
    mask = np.zeros(im_tmp.shape,dtype=np.uint8)
    im_tmp [minr:maxr,minc:maxc,:] = 255
    mask [minr:maxr,minc:maxc,:] = 255
    mask [x0,y0] = 0
    imageio.imwrite(outRootPath+'input_'+images[i], im_tmp)
    imageio.imwrite(outRootPath+'mask_'+images[i], mask)
print('done')
