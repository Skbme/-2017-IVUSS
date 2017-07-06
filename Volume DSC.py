import os
import sys
import scipy.misc
import Voxel as vx
import numpy as np
import natsort 		# For natural sorting
#from natsort import natsorted

DataDir = './Result/WALL/'


def Getraws(npyDir):
    # Read the images and concatenate
    images = []
    filenames = []
    for dirName, subdirList, fileList in os.walk(npyDir):
        # Sort the tif file numerically
        fileList = natsort.natsorted(fileList)
            
        for f in fileList:
            if f.endswith(".bin"):
                filename, file_extension = os.path.splitext(f)
                fullpath = os.path.join(npyDir,f)

                
                print filename

                # check!
                pVoxel = vx.PyVoxel()
                #pVoxel.ReadFromRaw(fullpath)
                pVoxel.ReadFromBin(fullpath) 
                #pVoxel.Normalize()

                images.append(pVoxel.m_Voxel)
                filenames.append(filename)

    return images, filenames


if __name__ == '__main__':
    
    DataDir = './Result/WALL/'
    GTDir = './Result/GT WALL/'
    
    images, filenames = Getraws(DataDir)
    images_GT, filenames_GT = Getraws(GTDir)

    images = np.array(images)
    images_GT = np.array(images_GT)

    DSCs = [0, 0]

    shape = images.shape
    for i in range(shape[0]):
        for j in range(2):
            print(j)
            Seg = (images[i] == (j))
            GT = (images_GT[i] == (j))

            Seg = np.array(Seg)
            GT = np.array(GT)

            Seg = Seg.astype(np.float32)
            GT = GT.astype(np.float32)

            Intersec = GT * Seg
         
            Sum = GT.sum() + Seg.sum()

            DSC = Intersec.sum()*2/Sum
            DSCs[j] = DSC
         
            print Intersec.sum()
            
            #print j+1
        print filenames[i], DSCs[0], DSCs[1]


    #data = my_to_categorical(data)
    #data = np.expand_dims(data, axis=3)

    #print data.shape

    #np.save('train_normalized.npy', data)
    #np.save('train_mask.npy', data)
    #np.save('train_mask_test_weight.npy', data)

    # data = np.load('train.npy')
    # mask = np.load('train_mask_test.npy')
    # pVoxel = vx.PyVoxel()
    # pVoxel.NumpyArraytoVoxel(data)
    # pVoxel.WriteToRaw('test_data.raw')
    # pVoxel.WriteToBin('mask.bin')

    # pVoxel.NumpyArraytoVoxel(mask)
    # pVoxel.WriteToBin('mask.bin')

    # pVoxel.ReadFromRaw("Test.raw")
    # #pVoxel.m_Voxel = pVoxel.AdjustPixelRange(0, 300)
    # pVoxel.WriteToRaw("Test2.raw")
    #p2.WriteToRaw("Test2.raw")

