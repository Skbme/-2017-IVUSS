import os
import sys
import scipy.misc
import Voxel as vx
import numpy as np
import natsort 		# For natural sorting
from natsort import natsorted

#DataDir = './Image/ED_Training/'
#DataDir = './Mask/ED_Training/'

# DataDir = './Image/ED_Test/'
#DataDir = './Mask/ED_Test/'
# DataDir = './Resized/Image/Training/'
DataDir = './Resized/MASK/LUMEN/Test/'
"""
IVUS

"""
# DataDir = 'F:\\Sekeun\\3__Data\\[2017]MICCAI PAPER\\IVUS_256\\IVUS_256\\Resized\\IMAGE\\Training'
# DataDir = 'F:\\Sekeun\\3__Data\\[2017]MICCAI PAPER\\IVUS_256\\IVUS_256\\Resized\\MASK\\LUMEN\\Training'
# DataDir = 'F:\\Sekeun\\3__Data\\[2017]MICCAI PAPER\\IVUS_256\\IVUS_256\\Resized\\MASK\\WALL\\Training'
def Getraws(npyDir):
    # Read the images and concatenate
    images = []
    filenames = []
    for dirName, subdirList, fileList in os.walk(npyDir):
        # Sort the tif file numerically
        # fileList = natsort.natsort(fileList)
        fileList = natsorted(fileList)
        for f in fileList:            
            if f.endswith(".bin"):
                filename, file_extension = os.path.splitext(f)
                fullpath = os.path.join(npyDir,f)
                # check!
                pVoxel = vx.PyVoxel()
                # pVoxel.ReadFromRaw(fullpath)
                pVoxel.ReadFromBin(fullpath) 
                # pVoxel.Normalize()

                images.append(pVoxel.m_Voxel)
                filenames.append(filename)
    return images, filenames

def my_to_categorical(y, nb_classes=None):
    Y = np.zeros([y.shape[0],y.shape[1],y.shape[2],y.max()+1],dtype='uint8')
    Y[np.nonzero(y)[0],np.nonzero(y)[1],np.nonzero(y)[2],y[np.nonzero(y)[0],np.nonzero(y)[1],np.nonzero(y)[2]]] = 2
    Y[np.where(y == 0)[0],np.where(y == 0)[1],np.where(y == 0)[2],y[np.where(y == 0)[0],np.where(y == 0)[1],np.where(y == 0)[2]]] = 1  
    return Y
    
if __name__ == '__main__':
    FullDir = DataDir
    images, filenames = Getraws(FullDir)
    print(filenames)


    images = np.array(images)
    data = images[0]
    # NumpyDataDir = os.path.join(FullDir, filenames[0])
    # print(NumpyDataDir)
    np.save(filenames[0]+'.npy', images[0])

    shape = images.shape
    for i in range(shape[0]-1):
        data = np.concatenate((data, images[i+1]), axis=0)
        #print images[i].max()
        # NumpyDataDir = os.path.join(FullDir, filenames[i+1])
        # np.save(NumpyDataDir+'.npy', images[i+1])
        # pVoxel = vx.PyVoxel()
        # pVoxel.NumpyArraytoVoxel(images[i+1])
        # pVoxel.WriteToRaw(filenames[i+1]+'.raw')
        # print (NumpyDataDir)



    data = my_to_categorical(data)
    #data = np.expand_dims(data, axis=3)

    print (data.shape)

    # np.save('F:\\Sekeun\\3__Data\\[2017]MICCAI PAPER\\IVUS_256\\IVUS_256\\train_normalized.npy', data)
    np.save('test_LUMEN mask.npy', data)

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

