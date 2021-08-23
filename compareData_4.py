import os
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

dataset_path = './datasets/demo/'
outputimg_path = './output_img/'

datasetimg = [i for i in os.listdir(dataset_path+'A/')]
datasetimg.sort()

outputimg = [j for j in os.listdir(outputimg_path)]
outputimg.sort()

if len(datasetimg) != len(outputimg):
    print("can't make compared_img: datasetimgs and outputimgs doesn't match")
else:
    if not os.path.exists('./compared_test'):
        os.mkdir('./compared_test')

    tp_rate = 0.5
    idx = 0
    for i in tqdm(range(len(datasetimg))):
        Aimage = Image.open(dataset_path+'A/' + datasetimg[i])
        Bimage = Image.open(dataset_path+'B/' + datasetimg[i])
        resultimg = Image.open(outputimg_path+outputimg[i])
        
        Aimage = np.array(Aimage)
        Bimage = np.array(Bimage)
        
        resultimg = np.array(resultimg, np.uint8)
        resultimg = np.expand_dims(resultimg, axis=2)
        resultconc1 = np.concatenate((resultimg,resultimg),axis=2)
        result3ch = np.concatenate((resultconc1,resultimg),axis=2)
        
        compare_img = tp_rate*result3ch[:,:,:] + (1 - tp_rate)*Bimage[:,:,:]
        compare_img = np.array(np.around(compare_img), np.uint8)
        final_result = np.concatenate((Aimage,Bimage,compare_img),axis=1)
        final_result_PIL = Image.fromarray(final_result)
        final_result_PIL.save('./compared_test/' + datasetimg[i])
