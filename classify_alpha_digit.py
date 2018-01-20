#coding=utf-8
import tensorflow as tf
import numpy as np
import os
from load_data import all_data_list_1_2,random_batch_data

checkpoint_dir="/home/jobs/Desktop/charNumReg/train_hand_model"

x_input=tf.placeholder(tf.float32,shape=(None,50,50,3))
y_input=tf.placeholder(tf.float32,shape=(None,62))
#y_initial=tf.placeholder(tf.int32,shape=(None))
#y_input=tf.one_hot(y_initial,62)

with tf.variable_scope("conv1") as scope:
	conv1_weight=tf.get_variable(name="conv1_weight",shape=[5,5,3,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
	conv1_biases=tf.get_variable(name="conv1_biases",shape=[32],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
	conv1=tf.nn.conv2d(x_input,conv1_weight,strides=[1,1,1,1],padding="SAME")
	relu1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))
	pool1=tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
with tf.variable_scope("conv2") as scope:
	conv2_weight=tf.get_variable(name="conv2_weight",shape=[5,5,32,64],initializer=tf.truncated_normal_initializer(stddev=0.1))
	conv2_biases=tf.get_variable(name="conv2_biases",shape=[64],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
	conv2=tf.nn.conv2d(pool1,conv2_weight,strides=[1,1,1,1],padding="SAME")
	relu2=tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))
	pool2=tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
with tf.variable_scope("conv3") as scope:
	conv3_weight=tf.get_variable(name="conv3_weight",shape=[5,5,64,128],initializer=tf.truncated_normal_initializer(stddev=0.1))
	conv3_biases=tf.get_variable(name="conv3_biases",shape=[128],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
	conv3=tf.nn.conv2d(pool2,conv3_weight,strides=[1,1,1,1],padding="SAME")
	relu3=tf.nn.relu(tf.nn.bias_add(conv3,conv3_biases))
	pool3=tf.nn.max_pool(relu3,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
pool3_shape=pool3.get_shape().as_list()
shape=pool3_shape[1]*pool3_shape[2]*pool3_shape[3]
#7*7*128
pool_flatten=tf.reshape(pool3,[-1,shape])
fc1_weight=tf.get_variable(name="fc1_weight",shape=[shape,1024],initializer=tf.truncated_normal_initializer(stddev=0.1))
fc1_biases=tf.get_variable(name="fc1_biases",shape=[1024],initializer=tf.constant_initializer(0.1))
fc1=tf.nn.relu(tf.matmul(pool_flatten,fc1_weight)+fc1_biases)
keep_prob=tf.placeholder(tf.float32)
fc1_dropout=tf.nn.dropout(fc1,keep_prob)


fc2_weight=tf.get_variable(name="fc2_weight",shape=[1024,62],initializer=tf.truncated_normal_initializer(stddev=0.1))
fc2_biases=tf.get_variable(name="fc2_biases",shape=[62],initializer=tf.constant_initializer(0.1))
fc2=tf.matmul(fc1_dropout,fc2_weight)+fc2_biases

y_conv=tf.nn.softmax(fc2)

train_writer = tf.summary.FileWriter("/home/jobs/Desktop/charNumReg" +'/train_hand')
'''
global_step = tf.Variable(0)  
learning_rate = tf.train.exponential_decay(1e-2,global_step,decay_steps=385,decay_rate=0.98,staircase=True) 
'''
#regulizers=tf.nn.l2_loss(fc1_weight)+tf.nn.l2_loss(fc2_weight)+tf.nn.l2_loss(conv1_weight)+tf.nn.l2_loss(conv2_weight)+tf.nn.l2_loss(conv3_weight)


loss=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=fc2))

params=tf.trainable_variables()
opt=tf.train.GradientDescentOptimizer(0.01)
gradients=tf.gradients(loss,params)
print(len(gradients),len(params))
clipped_gradients,norm=tf.clip_by_global_norm(gradients,clip_norm=0.8)
train_step=opt.apply_gradients(zip(clipped_gradients,params))
#loss+=5e-4*regulizers
tf.summary.scalar("loss",loss)


#train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
#train_step=tf.train.GradientDescentOptimizer(0.01).minimize(loss)

pred1=tf.argmax(y_input,1)
pred2=tf.argmax(y_conv,1)
correct_prediction=tf.cast(tf.equal(tf.argmax(y_input,1),tf.argmax(y_conv,1)),tf.float32)
accuracy=tf.reduce_mean(correct_prediction)
merged=tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)

print("lodding data......")
batches_initial_img,batches_initial_label=random_batch_data(all_data_list_1_2())
saver=tf.train.Saver()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	with open("/home/jobs/Desktop/charNumReg/acc_hand_.txt",'a') as f:
		for i in range(1000):
			batches_img,batches_label=batches_initial_img,batches_initial_label
			per_acc=0.0
			per_loss=0.0

			all_batch=len(batches_img)
			for i_ in range(len(batches_img)):
				summary,train_loss,train_acc,_=sess.run([merged,loss,accuracy,train_step],feed_dict={x_input:batches_img[i_],y_input:batches_label[i_],keep_prob:0.5})
				print(train_acc,train_loss)
				per_acc=per_acc+train_acc
				per_loss=per_loss+train_loss
			#train_writer.add_summary(summary, i)
			print("epoch: %d,		training_loss %g,		training_acc %g"%(i,per_loss/all_batch,per_acc/all_batch))
			f.write("epoch: %d,		training_loss %g,		training_acc %g"%(i,per_loss/all_batch,per_acc/all_batch)+"\n")
			f.flush()
			if i%50==0 and i!=0:
				saver.save(sess,checkpoint_dir+"/model.ckpt",global_step=i)
		train_writer.close()
