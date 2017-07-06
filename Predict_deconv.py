#!/usr/bin/env python



# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 10:12:52 2016

@author: tmquan
"""
import skimage.io
#from Model_deconv 		import *
#from Model_custom_loss 		import *  
#from Model_revised 		import *

from IVUS_MODEL_test2 		import *
#from Model_final_test_170412 import *
import Voxel as vx
from TFlearn	import *
from Utility	import *
import os
import tensorflow as tf
#import natsort 		# For natural sorting

#SaveDir = './Result/Unet-softmax/'
SaveDir = './Result/LUMEN/'

def Getnpys():
	# Read the images and concatenate
	images = []
	filenames = []
	npyDir = './Resized/IMAGE/Test/'
	for dirName, subdirList, fileList in os.walk(npyDir):
		# Sort the tif file numerically
		#fileList = natsort.natsort(fileList) 

		for f in fileList:
			if f.endswith(".npy"):
				filename, file_extension = os.path.splitext(f)
				fullpath = os.path.join(npyDir,f)

				print filename

				image = np.load(fullpath)
				images.append(image)
				filenames.append(filename)

	return images, filenames

			

def predict():
	# Load the model
	model = get_model()

	#model.load("Unet-softmax/model_ACDC")
	model.load("Model_StairNet_IVUS")

	Images, Filenames = Getnpys()

	numImage = len(Images)

	#for i in images
	for i in range(numImage):

		# get a pair of images
		img = Images[i]
		img = np.expand_dims(img, axis=3)

		print(Filenames[i])
		print(img.shape)

		# Deploy the model on img
		y_pred = model.predict(img.astype(np.float32))
		y_pred = np.array(y_pred).astype(np.float32)

		y_pred[y_pred < 0.5] = 0
		#y_pred = np.clip(y_pred, 0, 1)
		y_pred = np.argmax(y_pred, axis=3)
		shape = y_pred.shape
		#print(y_pred.shape)
		#y_pred = y_pred.reshape(shape[0], shape[1], shape[2])
	
		# y_pred = np.round(y_pred)
		# y_pred = np.clip(y_pred, 0, 1)
		# y_pred = np.reshape(y_pred, (-1, 256, 256))
		# y_pred = y_pred.astype(np.uint8, copy=False)
		print(y_pred.shape)

		# Save the prediction
		# idx_aorta = y_pred > 0
		# y_pred[idx_aorta] = 255
		# skimage.io.imsave(SaveDir + Filenames[i] + "_pred.tif", y_pred)

		##################################################################################
		# Make the overlay
		# Convert img from gray to 3 channels image and overlay the mask
		# img = np.squeeze(img)
		# rgbImg = np.zeros([img.shape[0],256,256,3], dtype=np.uint8)
		# rgbImg[:,:,:,0] = img
		# rgbImg[:,:,:,1] = img
		# rgbImg[:,:,:,2] = img

		# rgbImg[idx_aorta,0] = (rgbImg[idx_aorta,0] + 255) / 2
		# rgbImg[idx_aorta,1] = rgbImg[idx_aorta,1] / 2
		# rgbImg[idx_aorta,2] = rgbImg[idx_aorta,2] / 2
		
		# skimage.io.imsave(SaveDir + Filenames[i] + "_overlay.tif", rgbImg)

		np.save(SaveDir + Filenames[i] + "_pred.npy", y_pred)
		pVoxel = vx.PyVoxel()
		pVoxel.NumpyArraytoVoxel(y_pred)
		pVoxel.WriteToBin(SaveDir + Filenames[i] + "_pred.bin")

if __name__ == '__main__':
	import os
	os.environ["CUDA_VISIBLE_DEVICES"]="3"
	from tensorflow.python.client import device_lib
	print(device_lib.list_local_devices())

	#tf.device('/gpu:1')

	predict()
