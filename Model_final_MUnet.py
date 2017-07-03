import numpy
from TFlearn import *
import tensorflow as tf
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
		net 	= conv_2d(net, filter_count, kernel_size)
		if batch_norm:
			net = batch_normalization(net)
		if act_type != "":
			net = activation(net, act_type)

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

	net  = input_data(shape=[None, 3, 256, 256, 1],
					 data_preprocessing=Preprocessing,
					 data_augmentation=Augmentation)

	# Setting hyper parameter
	kernel_size 	= 3
	filter_count 	= 32	 # Original unet use 64 and 2 layers of conv

	net = tf.pad(net, [[0, 0], [0, 0], [3, 3], [3, 3], [0, 0]], "CONSTANT")
	net = conv_3d(net, filter_count/2, [3, 7, 7], padding='valid')
	net = batch_normalization(net)
	net = activation(net, "relu")

	shape = tflearn.utils.get_incoming_shape(net)
	print shape
	net = reshape(net, [-1, shape[2], shape[3], shape[4]])

	#net 	= net/255

	#[Con (3x3) * 64 -> Batch -> Act(Relu)] X 2 -> Down pooling
	net		= convolution_module(net, kernel_size, filter_count = filter_count * 1)
	layer64 = net;
	net 	= max_pool_2d(net, 2, strides=2)
	#net		= dropout(net, 0.5)


	# [Con (3x3) * 128 -> Batch -> Act(Relu)] X 2 -> Down pooling
	net		= convolution_module(net, kernel_size, filter_count = filter_count * 2)
	layer128 = net;
	net 	= max_pool_2d(net, 2, strides=2)
	#net		= dropout(net, 0.5)


	# [Con (3x3) * 256 -> Batch -> Act(Relu)] X 2 -> Down pooling
	net		= convolution_module(net, kernel_size, filter_count = filter_count * 4)
	layer256 = net;
	net 	= max_pool_2d(net, 2, strides=2)
	#net		= dropout(net, 0.5)


	# [Con (3x3) * 512 -> Batch -> Act(Relu)] X 2 -> Down pooling
	net 	= convolution_module(net, kernel_size, filter_count = filter_count * 8)
	layer512 = net;
	net 	= max_pool_2d(net, 2, strides=2)
	net 	= dropout(net, 0.5)


	# [Con (3x3) * 1024 -> Batch -> Act(Relu)] X 2
	net		= convolution_module(net, kernel_size, filter_count = filter_count * 16)
	net		= dropout(net, 0.5)


	# Con (1x1) * 512 -> Upsampling(2x2) [-> Batch -> Act(Relu)]
	net = convolution_module(net, 2, filter_count = filter_count * 8, up_pool=True, convolution=False)

	# Merge 512 + 512 = 1024 -> [Con (3x3) * 512 -> Batch -> Act(Relu)] X 2
	net		= merge([layer512, net], mode='concat', axis=3)
	net 	= convolution_module(net, kernel_size, filter_count = filter_count * 8)
	#net 	= dropout(net, 0.5)


	# Con (1x1) * 256 -> Upsampling(2x2) [-> Batch -> Act(Relu)]
	net = convolution_module(net, 2, filter_count = filter_count * 4, up_pool=True, convolution=False)

	# Merge 256 + 256 = 512 -> [Con (3x3) * 256 -> Batch -> Act(Relu)] X 2
	net		= merge([layer256, net], mode='concat', axis=3)
	net 	= convolution_module(net, kernel_size, filter_count = filter_count * 4)
	#net 	= dropout(net, 0.5)


	# Con (1x1) * 128 -> Upsampling(2x2) [-> Batch -> Act(Relu)]
	net		= convolution_module(net, 2, filter_count = filter_count * 2, up_pool=True, convolution=False)

	# Merge 128 + 128 = 256 -> [Con (3x3) * 128 -> Batch -> Act(Relu)] X 2
	net 	= merge([layer128, net], mode='concat', axis=3)
	net 	= convolution_module(net, kernel_size, filter_count = filter_count * 2)
	#net 	= dropout(net, 0.5)


	# Con (1x1) * 64 -> Upsampling(2x2) [-> Batch -> Act(Relu)]
	net 	= convolution_module(net, 2, filter_count = filter_count * 1, up_pool=True, convolution=False)

	# Merge 64 + 64 = 128 -> [Con (3x3) * 64 -> Batch -> Act(Relu)] X 2
	net 	= merge([layer64, net], mode='concat', axis=3)
	net 	= convolution_module(net, kernel_size, filter_count = filter_count * 1)
	#net 	= dropout(net, 0.5)

	net 	= conv_2d(net, 4, 1)
	#net 	= batch_normalization(net)
	#net		= activation(net, "sigmoid")
	
	#Rank problem
	#net = tf.nn.softmax(net)


	# To avoid numerical unstability, normalize it by subtracting 
	pwMax = tf.reduce_max(net, 3, keep_dims=True)
	pwMax = tf.tile(pwMax, tf.pack([1, 1, 1, tf.shape(net)[3]]))
	net = net - pwMax

	exponential_map = tf.exp(net)
	sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
	tensor_sum_exp = tf.tile(sum_exp, tf.pack([1, 1, 1, tf.shape(net)[3]]))
	net = tf.div(exponential_map,tensor_sum_exp)


	# net = convolution_module(net, kernel_size, filter_count=16, batch_norm=False) #Relu
	# net     = tf.cast(net > 0.5, tf.float32)
	"""
	Define the architecture of the network is here
	"""
	# net 	= highway_conv_2d(net, 8, 3, activation='sigmoid')
	# net 	= highway_conv_2d(net, 2, 1, activation='sigmoid')
	return net
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
					 
	# Training the network
	model = DNN(net, 
				checkpoint_path='models',
				tensorboard_verbose=3)
	return model

