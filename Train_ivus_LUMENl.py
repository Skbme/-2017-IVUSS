from IVUS_MODEL_test2 		import *
from TFlearn	import *
from Utility	import *

def train():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    from tensorflow.python.client import device_lib
    import tensorflow as tf
    from TFlearn	import *
	
    #X = np.load('train.npy')
    X = np.load('train_normalized.npy')
    y = np.load('train_LUMEN mask.npy')
    # y = np.load('train_mask_test_weight.npy')
    print X.shape
    print y.shape
    
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    # Get the model from Model.py
    model = get_model()
    # Shuffle the data
    print('Shuffle data...')
    X, y = shuffle(X, y)

    model.fit(X, y, run_id="IVUS_sk",
              n_epoch=50,
              validation_set=0.1,
              shuffle=True,
              show_metric=True,
              snapshot_epoch=True,
              batch_size=20)
    model.save('Model_StairNet_IVUS_LUMEN')    


if __name__ == '__main__':
	train()
