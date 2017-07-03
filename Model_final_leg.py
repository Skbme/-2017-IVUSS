import numpy
import tensorflow as tf
#from tensorflow import variables_initializer
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
	filter_count 	= 16	 # Original unet use 64 and 2 layers of conv

	LLeg2 = max_pool_2d(X, 2, strides=2) # 128 x 128
	LLeg3 = max_pool_2d(LLeg2, 2, strides=2)
	LLeg4 = max_pool_2d(LLeg3, 2, strides=2)
	LLeg5 = max_pool_2d(LLeg4, 2, strides=2)

	#net 	= net/255

	# layer 1 256 x 256
	layer1_1 = conv_2d(X, filter_count, kernel_size)
	layer1_1 = batch_normalization(layer1_1)
	layer1_1 = activation(layer1_1, 'relu')

	layer1_2 = merge([layer1_1, X], mode='concat', axis=3)
	
	layer1_2 = conv_2d(layer1_2, filter_count * 2, kernel_size)
	layer1_2 = batch_normalization(layer1_2)
	layer1_2 = activation(layer1_2, 'relu')

	layer1_p 	= max_pool_2d(layer1_2, 2, strides=2) # 128 x 128

	# layer 2 128 x 128
	layer2_1 = merge([layer1_p, LLeg2], mode='concat', axis=3)

	layer2_1 = conv_2d(layer2_1, filter_count * 2, kernel_size)
	layer2_1 = batch_normalization(layer2_1)
	layer2_1 = activation(layer2_1, 'relu')

	layer2_1 = merge([layer2_1, layer1_p, LLeg2], mode='concat', axis=3)

	layer2_2 = conv_2d(layer2_1, filter_count * 3, kernel_size)
	layer2_2 = batch_normalization(layer2_2)
	layer2_2 = activation(layer2_2, 'relu')

	layer2_p 	= max_pool_2d(layer2_2, 2, strides=2) # 64 x 64

	# layer 3 64 x 64
	layer3_1 = merge([layer2_p, LLeg3], mode='concat', axis=3)

	layer3_1 = conv_2d(layer3_1, filter_count * 3, kernel_size)
	layer3_1 = batch_normalization(layer3_1)
	layer3_1 = activation(layer3_1, 'relu')

	layer3_1 = merge([layer3_1, layer2_p, LLeg3], mode='concat', axis=3)

	layer3_2 = conv_2d(layer3_1, filter_count * 4, kernel_size)
	layer3_2 = batch_normalization(layer3_2)
	layer3_2 = activation(layer3_2, 'relu')
	layer3_2 	= dropout(layer3_2, 0.5)

	layer3_p 	= max_pool_2d(layer3_2, 2, strides=2) # 32 x 32


	# layer4 32 x 32
	layer4_1 = merge([layer3_p, LLeg4], mode='concat', axis=3)

	layer4_1 = conv_2d(layer4_1, filter_count * 4, kernel_size)
	layer4_1 = batch_normalization(layer4_1)
	layer4_1 = activation(layer4_1, 'relu')

	layer4_1 = merge([layer4_1, layer3_p, LLeg4], mode='concat', axis=3)

	layer4_2 = conv_2d(layer4_1, filter_count * 8, kernel_size)
	layer4_2 = batch_normalization(layer4_2)
	layer4_2 = activation(layer4_2, 'relu')

	layer4_3 = conv_2d(layer4_2, filter_count * 4, kernel_size)
	layer4_3 = batch_normalization(layer4_3)
	layer4_3 = activation(layer4_3, 'relu')
	layer4_3 	= dropout(layer4_3, 0.5)

	# layer 3.2 64 x 64
	layer33_u = upsample_2d(layer4_3, 2)

	layer33_1 = merge([layer33_u, layer3_2, LLeg3], mode='concat', axis=3)

	layer33_1 = conv_2d(layer33_1, filter_count * 4, kernel_size)
	layer33_1 = batch_normalization(layer33_1)
	layer33_1 = activation(layer33_1, 'relu')

	layer33_2 = merge([layer33_1, layer33_u, layer3_2, LLeg3], mode='concat', axis=3)

	layer33_2 = conv_2d(layer33_2, filter_count * 3, kernel_size)
	layer33_2 = batch_normalization(layer33_2)
	layer33_2 = activation(layer33_2, 'relu')

	# layer 2.2 128 x 128
	layer22_u = upsample_2d(layer33_2, 2)

	layer22_1 = merge([layer22_u, layer2_2, LLeg2], mode='concat', axis=3)

	layer22_1 = conv_2d(layer22_1, filter_count * 3, kernel_size)
	layer22_1 = batch_normalization(layer22_1)
	layer22_1 = activation(layer22_1, 'relu')

	layer22_2 = merge([layer22_1, layer22_u, layer2_2, LLeg2], mode='concat', axis=3)

	layer22_2 = conv_2d(layer22_2, filter_count * 2, kernel_size)
	layer22_2 = batch_normalization(layer22_2)
	layer22_2 = activation(layer22_2, 'relu')


	# layer 1.1 256 x 256
	layer11_u = upsample_2d(layer22_2, 2)

	layer11_1 = merge([layer11_u, layer1_2, X], mode='concat', axis=3)

	layer11_1 = conv_2d(layer11_1, filter_count * 2, kernel_size)
	layer11_1 = batch_normalization(layer11_1)
	layer11_1 = activation(layer11_1, 'relu')

	layer11_2 = merge([layer11_1, layer11_u, layer1_2, X], mode='concat', axis=3)

	layer11_2 = conv_2d(layer11_2, filter_count, kernel_size)
	layer11_2 = batch_normalization(layer11_2)
	layer11_2 = activation(layer11_2, 'relu')


	# last
	RLeg2 = merge([layer22_u, layer2_2, LLeg2], mode='concat', axis=3)
	RLeg2_u = upsample_2d(RLeg2, 2)

	RLeg3 = merge([layer33_u, layer3_2, LLeg3], mode='concat', axis=3)
	RLeg3_u = upsample_2d(RLeg3, 4)

	final = merge([layer11_2, layer11_u, RLeg2_u, RLeg3_u, layer1_2, X], mode='concat', axis=3)
	#final = conv_2d(final, 200, kernel_size)
	final = conv_2d(final, 64, kernel_size)
	final = batch_normalization(final)
	final = activation(final, 'relu')

	final 	= conv_2d(final, 4, 1)

	# To avoid numerical unstability, normalize it by subtracting 
	pwMax = tf.reduce_max(final, 3, keep_dims=True)
	pwMax = tf.tile(pwMax, tf.pack([1, 1, 1, tf.shape(final)[3]]))
	final = final - pwMax

	exponential_map = tf.exp(final)
	sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
	tensor_sum_exp = tf.tile(sum_exp, tf.pack([1, 1, 1, tf.shape(final)[3]]))
	final = tf.div(exponential_map,tensor_sum_exp)


	# net = convolution_module(net, kernel_size, filter_count=16, batch_norm=False) #Relu
	# net     = tf.cast(net > 0.5, tf.float32)
	"""
	Define the architecture of the network is here
	"""
	# net 	= highway_conv_2d(net, 8, 3, activation='sigmoid')
	# net 	= highway_conv_2d(net, 2, 1, activation='sigmoid')
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
				checkpoint_path='ES_Sub4_models',
				tensorboard_verbose=0)

	return model

