import os
import sys
import scipy.misc
import Voxel as vx
import numpy as np
import natsort 		# For natural sorting
#from natsort import natsorted

#DataDir = './Image/ED_Training/'
#DataDir = './Image2/ED_Test/'
#DataDir = './Mask/ED_Training/'

#DataDir = './Image/ED_Test/'
#DataDir = './Mask/ED_Test/'

# DataDir1 = './Image3/ED_Training/'
# DataDir2 = './Image3/ES_Training/'
DataDir1 = './Image3/ED_Test/'
DataDir2 = './Image3/ES_Test/'
#DataDir = './Mask/ES_Training/'

# DataDir1 = './Mask/ED_Training/'
# DataDir2 = './Mask/ES_Training/'


def Getraws(npyDir):
    # Read the images and concatenate
    images = []
    filenames = []
    for dirName, subdirList, fileList in os.walk(npyDir):
        # Sort the tif file numerically
        fileList = natsort.natsort(fileList)
            
        for f in fileList:
            if f.endswith(".raw"):
                filename, file_extension = os.path.splitext(f)
                fullpath = os.path.join(npyDir,f)

                print filename

                # check!
                pVoxel = vx.PyVoxel()
                pVoxel.ReadFromRaw(fullpath)
                #pVoxel.ReadFromBin(fullpath) 
                #pVoxel.Normalize()
                pVoxel.NormalizeMM()
                #pVoxel.AdjustPixelRangeNormalize(170)

                pVoxel.m_Voxel = np.expand_dims(pVoxel.m_Voxel, axis=3)
                pVoxel.m_Voxel = np.transpose(pVoxel.m_Voxel, (0, 3, 1, 2))

                images.append(pVoxel.m_Voxel)
                filenames.append(filename)
    return images, filenames

def my_to_categorical(y, nb_classes=None):
    Y = np.zeros([y.shape[0],y.shape[1],y.shape[2],y.max()+1],dtype='uint8')
    Y[np.nonzero(y)[0],np.nonzero(y)[1],np.nonzero(y)[2],y[np.nonzero(y)[0],np.nonzero(y)[1],np.nonzero(y)[2]]] = 8
    Y[np.where(y == 0)[0],np.where(y == 0)[1],np.where(y == 0)[2],y[np.where(y == 0)[0],np.where(y == 0)[1],np.where(y == 0)[2]]] = 1  
    return Y
    
# if __name__ == '__main__':
#     #FullDir = DataDir
#     #images, filenames = Getraws(FullDir)
#     images, filenames = Getraws(DataDir1)
#     images2, filenames2 = Getraws(DataDir2)

#     images = np.array(images)
#     #np.save(filenames[0]+'.npy', images[0])
#     shape = images.shape

#     dim = images[0].shape
#     data2 = images[0][0]
#     data2 = np.concatenate((data2, images[0][1]), axis=0)
#     data2 = np.concatenate((data2, images[0][2]), axis=0)

#     data = np.expand_dims(data2, axis=3)
#     data = np.transpose(data, (3, 0, 1, 2))

#     for i in range(dim[0]-3):
#         idx = i+2
#         data2 = images[0][idx-1]
#         data2 = np.concatenate((data2, images[0][idx]), axis=0)
#         data2 = np.concatenate((data2, images[0][idx+1]), axis=0)

#         data2 = np.expand_dims(data2, axis=3)
#         data2 = np.transpose(data2, (3, 0, 1, 2))
#         data = np.concatenate((data, data2), axis=0)
    
#     for i in range(shape[0]-1):
#         idx = i + 1
#         dim = images[idx].shape
#         for j in range(dim[0]-2):
#             data2 = images[idx][j]
#             data2 = np.concatenate((data2, images[idx][j+1]), axis=0)
#             data2 = np.concatenate((data2, images[idx][j+2]), axis=0)

#             data2 = np.expand_dims(data2, axis=3)
#             data2 = np.transpose(data2, (3, 0, 1, 2))
#             data = np.concatenate((data, data2), axis=0)

#     images2 = np.array(images2)
#     #np.save(filenames[0]+'.npy', images[0])
#     shape = images2.shape

#     for i in range(shape[0]):
#         idx = i
#         dim = images[idx].shape
#         for j in range(dim[0]-2):
#             data2 = images[idx][j]
#             data2 = np.concatenate((data2, images[idx][j+1]), axis=0)
#             data2 = np.concatenate((data2, images[idx][j+2]), axis=0)

#             data2 = np.expand_dims(data2, axis=3)
#             data2 = np.transpose(data2, (3, 0, 1, 2))
#             data = np.concatenate((data, data2), axis=0)
    
  
#         #print images[i].max()
#         #np.save(filenames[i+1]+'.npy', images[i+1])
#         # pVoxel = vx.PyVoxel()
#         # pVoxel.NumpyArraytoVoxel(images[i+1])
#         # pVoxel.WriteToRaw(filenames[i+1]+'.raw')

#     # images2 = np.array(images2)
#     # shape = images2.shape
#     # for i in range(shape[0]-1):
#     #     data = np.concatenate((data, images2[i+1]), axis=0)
#     #     #print images[i].max()
#     #     #np.save(filenames[i+1]+'.npy', images[i+1])
#     #     # pVoxel = vx.PyVoxel()
#     #     # pVoxel.NumpyArraytoVoxel(images[i+1])
#     #     # pVoxel.WriteToRaw(filenames[i+1]+'.raw')
#     #     print filenames2[i+1]


#     #data = my_to_categorical(data)
#     #data = np.expand_dims(data, axis=3)

#     print data.shape

#     np.save('trainEDES_normalizedMM_MUnet.npy', data)

if __name__ == '__main__':

    images, filenames = Getraws(DataDir1)
    images2, filenames2 = Getraws(DataDir2)

    images = np.array(images)
    #np.save(filenames[0]+'.npy', images[0])
    shape = images.shape

    # for i in range(shape[0]):
    #     dim = images[i].shape

    #     data2 = images[i][0]
    #     data2 = np.concatenate((data2, images[i][1]), axis=0)
    #     data2 = np.concatenate((data2, images[i][2]), axis=0)
        
    #     data = np.expand_dims(data2, axis=3)
    #     data = np.transpose(data, (3, 0, 1, 2))    
    #     for j in range(dim[0]-3):
    #         idx = j + 1
    #         data2 = images[i][idx]
    #         data2 = np.concatenate((data2, images[i][idx+1]), axis=0)
    #         data2 = np.concatenate((data2, images[i][idx+2]), axis=0)

    #         data2 = np.expand_dims(data2, axis=3)
    #         data2 = np.transpose(data2, (3, 0, 1, 2))
    #         data = np.concatenate((data, data2), axis=0)
        
    #     np.save(filenames[i]+'.npy', data)
    #     print data.shape

    images2 = np.array(images2)
    #np.save(filenames[0]+'.npy', images[0])
    shape = images2.shape

    for i in range(shape[0]):
        dim = images2[i].shape

        data2 = images2[i][0]
        data2 = np.concatenate((data2, images2[i][1]), axis=0)
        data2 = np.concatenate((data2, images2[i][2]), axis=0)
        
        data = np.expand_dims(data2, axis=3)
        data = np.transpose(data, (3, 0, 1, 2))    
        for j in range(dim[0]-3):
            idx = j + 1
            data2 = images2[i][idx]
            data2 = np.concatenate((data2, images2[i][idx+1]), axis=0)
            data2 = np.concatenate((data2, images2[i][idx+2]), axis=0)

            data2 = np.expand_dims(data2, axis=3)
            data2 = np.transpose(data2, (3, 0, 1, 2))
            data = np.concatenate((data, data2), axis=0)
        
        np.save(filenames2[i]+'.npy', data)
        print data.shape
