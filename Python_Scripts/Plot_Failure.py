from IPython.display import Image, display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read the train file
df = pd.read_csv('./Data/train.csv')

ImageId = sys.argv[1]

image = ['./Data/train_images/'+ ImageId]

# Get Values from *.jpg
#ClassId = df.ClassId[iii]
#EncodedPixels = str(df.EncodedPixels[iii])


# Extract Pixel and Length From String
#EncodedPixels = EncodedPixels.split()
#pixel = []
#length = []
#df_failure = pd.DataFrame()
#
## Make lists for pixel and pixel-length
#for i in range(0,len(EncodedPixels),2):
#    pixel.append(int(EncodedPixels[i]))
#    length.append(int(EncodedPixels[i+1]))
#    
## Put Pixel and Length to a new DF    
#df_failure['Pixel'] = pixel
#df_failure['Length'] = length
#
## Grab pic boundaries
#img = mpimg.imread(listOfImageNames[0])
#height, width, channels = img.shape
#
## Figure as subplots
#fig, ax = plt.subplots(2, 1, constrained_layout=True, figsize=(25, 10))
#
## Subplot 1
#plt.subplot(2, 1, 1)
#plt.imshow(img)
#plt.xlim(0,1600)
#plt.ylim(0,256)
#plt.xticks(fontsize=16)
#plt.yticks(fontsize=16)
#ax[0].set_title('Picture Without Failure-Marking', fontsize= 24)
#ax[0].set_xlabel('Pixel in x', fontsize= 20)
#ax[0].set_ylabel('Pixel in y', fontsize= 20)
#
## Subplot 2
#plt.subplot(2, 1, 2)
#plt.imshow(img)
#plt.xlim(0,1600) 
#plt.ylim(0,256)
#plt.xticks(fontsize=16)
#plt.yticks(fontsize=16)
#ax[1].set_title('Picture With Failure-Marking, Failure Class: '+ str(ClassId), fontsize= 24)
#ax[1].set_xlabel('Pixel in x', fontsize= 20)
#ax[1].set_ylabel('Pixel in y', fontsize= 20)
# 
#
##preserve coordinate lists
#x = []
#y = []

# Loop for change pixel to coordiantes
#for ii in range(len(pixel)):
#    NumberOfColums = df_failure.Pixel[ii] // height
#    RemainPixelNo = (df_failure.Pixel[ii] / height - NumberOfColums) * height
#    x_coordinate = NumberOfColums + 1
#    y_coordinate = RemainPixelNo
#    
#    # Loop for generating pixels after start-pixel
#    for i in range(df_failure.Length[ii]-1):
#        
#        # Checkpoint whether end rows, if so jump in next column..
#        if y_coordinate + i > 256:
#            x.append(x_coordinate + 1)
#            y.append(i-(i-1))
#        else:
#            x.append(x_coordinate)
#            y.append(y_coordinate + i)
#
## Plot the failure pixels            
#plt.scatter(x, y, s=30, c='red')
#plt.show()
#
#fig.savefig('./Data/FailurePictures/'+str(ClassId)+'_FailureClass_'  + ImageId)