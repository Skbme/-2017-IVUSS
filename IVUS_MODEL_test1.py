import numpy
import tensorflow as tf
from TFlearn import *
import tflearn
from tflearn import utils
from tflearn import initializations

########################################################
# Real-time data preprocessing
Preprocessing = ImagePreprocessing()
Preprocessing.add_samplewise_zero_center() # revised 2017.01.20
Preprocessing.add_samplewise_stdnorm()
# Preprocessing.add_featurewise_zero_center()
# Preprocessing.add_featurewise_stdnorm()

# Real-time data augmentation
Augmentation  = ImageAugmentation()
Augmentation.add_random_blur()
def convolution_module(net, kernel_size, filter_count, batch_norm=True, up_pool=False, act_type="relu", convolution=True):
	import tflearn

	if up_pool:
		net = upsample_2d(net, kernel_size)
		net = conv_2d(net, filter_count, kernel_size)
		# if batch_norm:
		# 	net = batch_normalization(net)
		if act_type != "":
			net = activation(net, act_type)

	if convolution:
		post_net 	= conv_2d(net, filter_count, kernel_size)
		if batch_norm:
			post_net = batch_normalization(post_net)
		if act_type != "":
			post_net = activation(post_net, act_type)

		net = conv_2d(net, filter_count, kernel_size)
		if batch_norm:
			net = batch_normalization(net)
		if act_type != "":
			net = activation(net, act_type)

	shape = tflearn.utils.get_incoming_shape(net)
	print (shape)

	return net

def get_fcn():
	batch_size = 20 #shape[0] #20

	X = input_data(shape=[None, 256, 256, 1],
					 data_preprocessing=Preprocessing,
					 data_augmentation=Augmentation)
	
	# Setting hyper parameter
	kernel_size 	= 3
	filter_count 	= 16	 

    # layer 1 256 x 256
	L1 = conv_2d(X, filter_count, kernel_size)
	L1 = max_pool_2d(L1, 2, strides=2) 
	L1 = activation(L1, 'relu')	
	# UL1 = upsample_2d(L1, 2)
	# UL1 = conv_2d(UL1, filter_count * 2, kernel_size)
	# UL1 = activation(UL1, 'relu')
  
	L2 = conv_2d(L1, filter_count * 2, kernel_size)
	L2 = max_pool_2d(L2, 2, strides=2) 
	L2 = activation(L2, 'relu')	
	# SUL2 = upsample_2d(L2, 2)
	# SUL2 = conv_2d(SUL2, filter_count * 2, kernel_size)
	# SUL2 = activation(SUL2, 'relu')
	# UL2 = upsample_2d(SUL2, 2)
	# UL2 = conv_2d(UL2, filter_count * 4, kernel_size)
	# UL2 = activation(UL2, 'relu')

	L3 = conv_2d(L2, filter_count * 4, kernel_size)
	L3 = max_pool_2d(L3, 2, strides=2)
	L3 = activation(L3, 'relu')
	# SSUL3 = upsample_2d(L3, 2)
	# SSUL3 = conv_2d(SSUL3, filter_count * 8, kernel_size)
	# SSUL3 = activation(SSUL3, 'relu')
	# SUL3 = upsample_2d(SSUL3, 2)
	# SUL3 = conv_2d(SUL3, filter_count * 16, kernel_size)
	# SUL3 = activation(SUL3, 'relu')
	# UL3 = upsample_2d(SUL3, 2)
	# UL3 = conv_2d(UL3, filter_count * 32, kernel_size)
	# UL3 = activation(UL3, 'relu')

	L4 = conv_2d(L3, filter_count * 8, kernel_size)
	L4 = max_pool_2d(L4, 2, strides=2)
	L4 = activation(L4, 'relu')
	SSSUL4 = upsample_2d(L4, 2)
	SSSUL4 = conv_2d(SSSUL4, filter_count * 16, kernel_size)
	SSSUL4 = activation(SSSUL4, 'relu')
	SSUL4 = upsample_2d(SSSUL4, 2)
	SSUL4 = conv_2d(SSUL4, filter_count * 32, kernel_size)
	SSUL4 = activation(SSUL4, 'relu')
	SUL4 = upsample_2d(SSUL4, 2)
	SUL4 = conv_2d(SUL4, filter_count * 64, kernel_size)
	SUL4 = activation(SUL4, 'relu')
	UL4 = upsample_2d(SUL4, 2)
	UL4 = conv_2d(UL4, filter_count * 128, kernel_size)
	UL4 = activation(UL4, 'relu')

	# MergeLayer = merge([UL1, UL2, UL3, UL4, X], mode='concat', axis=3)
	final = conv_2d(UL4, 64, kernel_size)
	# final = batch_normalization(final)
	final = activation(final, 'relu')
	final = conv_2d(final, 64, 1)
	final = activation(final, 'relu')

	final = conv_2d(final, 64, 1)
	final = activation(final, 'relu')

	final = conv_2d(final, 2, 1)
	
	pwMax = tf.reduce_max(final, 3, keep_dims=True)
	pwMax = tf.tile(pwMax, tf.pack([1, 1, 1, tf.shape(final)[3]]))
	final = final - pwMax

	exponential_map = tf.exp(final)
	sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
	tensor_sum_exp = tf.tile(sum_exp, tf.pack([1, 1, 1, tf.shape(final)[3]]))
	final = tf.div(exponential_map,tensor_sum_exp)

	
	return final
########################################################
def get_model():
	net = get_fcn()

	# net = regression(net, 
	# 				 optimizer='adam', 
	# 				 learning_rate=0.001,
	#                  loss='mean_square') 
	# net = regression(net, 
	# 				learning_rate=0.001,
	# 				loss='weak_cross_entropy_2d_loss')
	def custom_acc(prediction, target, inputs):
		acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, 3), tf.argmax(target, 3)), tf.float32), name='acc')
		return acc
	def custom_loss(y_pred, y_true):

		old_shape = tflearn.utils.get_incoming_shape(y_pred)
		new_shape = [old_shape[0]*old_shape[1]*old_shape[2], old_shape[3]]
		cur_shape = [old_shape[0]*old_shape[1]*old_shape[2]*old_shape[3]]
		print (new_shape)
		# epsilon   = tf.constant(value=0.0001, shape=old_shape)
		# y_pred = y_pred + epsilon
		y_pred = tf.reshape(y_pred, new_shape)
		y_true = tf.reshape(y_true, new_shape)
		
		# y_pred = tf.nn.log_softmax(y_pred)
		#softmax_categorical_crossentropy, categorical_crossentropy, binary_crossentropy, mean_square, hinge_loss
		#http://tflearn.org/activations/
		# return tf.nn.sparse_softmax_cross_entropy_with_logits(y_pred, tf.to_int32(y_true))
  		# return tflearn.objectives.categorical_crossentropy(y_pred, y_true)
  		# return tflearn.objectives.roc_auc_score(y_pred, y_true)
  		# return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_pred, y_true))
  		# return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred, tf.to_int32(y_true)))

		with tf.name_scope('loss'):
			num_classes = y_true.get_shape()[-1]
			y_pred = tf.reshape(y_pred, new_shape)
			# shape = [y_pred.get_shape()[0], num_classes]
			epsilon = tf.constant(value=0.0001, shape=new_shape)
			y_pred = y_pred + epsilon
			y_true = tf.to_float(tf.reshape(y_true, new_shape))
			softmax = tf.nn.softmax(y_pred)

			cross_entropy = -tf.reduce_sum(y_true * tf.log(softmax), reduction_indices=[1])
			
			cross_entropy_mean = tf.reduce_mean(cross_entropy,
												name='xentropy_mean')
			tf.add_to_collection('losses', cross_entropy_mean)
			
			loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
		return cross_entropy_mean
		# return loss
		# return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_pred, y_true))

  		# return tflearn.objectives.weak_cross_entropy_2d(y_pred, tf.to_int32(y_true), num_classes=2)

	def custom_loss2(y_pred, y_true):
		# version 1
		# num_classes = 2

		# old_shape = y_pred.get_shape()
		# new_shape = [-1, num_classes]
		
		# y_pred = tf.reshape(y_pred, new_shape)
		# y_true = tf.reshape(y_true, new_shape)

		# with tf.name_scope('loss'):
		# 	y_pred_sum = tf.reduce_sum(y_pred, 1, keep_dims=True)
		# 	y_pred_sum = tf.tile(y_pred_sum, tf.pack([1, num_classes]))
		# 	y_pred_sum = tf.div(y_pred, y_pred_sum)
		# 	y_pred_sum = tf.clip_by_value(y_pred_sum,1e-10,1.0)

		# 	cross_entropy = -tf.reduce_sum(y_true * tf.log(y_pred_sum), reduction_indices=[1])
			
		# 	cross_entropy_mean = tf.reduce_mean(cross_entropy,
		# 										name='xentropy_mean_sigmoid')
		# 	tf.add_to_collection('losses', cross_entropy_mean)
			
		# 	loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
		# return cross_entropy_mean

		# version 2
		old_shape = tflearn.utils.get_incoming_shape(y_pred)
		num_classes = 2
		num_classes = old_shape[3]

		new_shape = [-1, num_classes]
		
		y_pred = tf.reshape(y_pred, new_shape)
		y_true = tf.reshape(y_true, new_shape)

		with tf.name_scope('loss'):
			# y_pred = tf.clip_by_value(y_pred,1e-10,1.0)
			y_pred = y_pred + 1e-8

			#cross_entropy = -tf.reduce_sum(y_true * tf.log(y_pred), reduction_indices=[1])
			cross_entropy = -tf.reduce_sum(y_true * tf.log(y_pred), 1)
		
			cross_entropy_mean = tf.reduce_mean(cross_entropy,
												name='xentropy_mean_sigmoid')
			tf.add_to_collection('losses', cross_entropy_mean)
			
			loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
		return cross_entropy_mean

	net = regression(net,
					 optimizer='RMSprop',
					 #optimizer='Adam',
					 # optimizer='AdaDelta',
					 # optimizer='SGD',
					 learning_rate=0.001,
					 # metric = tflearn.metrics.R2(),
					 # metric='Accuracy',
					 metric=custom_acc,
	                 # loss='binary_crossentropy') # categorical_crossentropy, binary_crossentropy, mean_square, hinge_loss
	                 #loss='categorical_crossentropy') #
	                 # loss='hinge_loss') # won't work
	                 # loss='mean_square') # won't work
	                 # loss='L2') #softmax_categorical_crossentropy, categorical_crossentropy, binary_crossentropy, mean_square, hinge_loss
	                 # loss='weak_cross_entropy_2d')
	                 loss=custom_loss2)
	
	# config = tf.ConfigProto()
	# # config.gpu_options.per_process_gpu_memory_fraction = 0.4
	# config.gpu_options.allow_growth = True
	# session = tf.Session(config=config)
	# session.run(tf.initialize_all_variables())

	# Training the network
	model = DNN(net, 
				checkpoint_path='models',
				tensorboard_verbose=3)

	return model

