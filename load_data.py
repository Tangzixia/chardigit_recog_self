#coding=utf-8
import os
import numpy as np
from scipy.misc import imread,imresize
import random
import cv2

def to_categorial(label,num_classes=62):
	len_=len(label)
	arr=np.zeros((len_,num_classes))
	for i in range(len(label)):
		arr[i,label[i]]=1.0
	return arr
def all_data_list_1_2(dir="/home/jobs/Downloads/English (3)/Fnt"):
	dirs= [[x[0],x[2]] for x in os.walk(dir)]
	dirs.remove(dirs[0])
	items_list=[]
	for item in dirs:
		dir=item[0]
		files=item[1]
		for file in files:
			path=os.path.join(dir,file)
			img=imresize(imread(path),[50,50])
			if img.shape!=(50,50,3):
				img=imresize(cv2.cvtColor( img, cv2.COLOR_GRAY2RGB ),[50,50])
			label=int(file.split("-")[0].split('img0')[1])-1
			items_list.append([img,label])
	return items_list
			
def random_batch_data(items_list,batch_size=20):
	batches_img=[]
	batches_label=[]
	len_=len(items_list)
	random.shuffle(items_list)
	batch=len_//batch_size

	count=0
	batch_img=[]
	batch_label=[]
	for item in items_list:
		if count%batch_size==0 and count!=0:
			batches_img.append(batch_img)
			batch_label=to_categorial(batch_label)
			batches_label.append(batch_label)
			batch_img=[]
			batch_label=[]
		batch_img.append(item[0]/255.)
		batch_label.append(item[1])
		count=count+1
	return batches_img,batches_label

if __name__=="__main__":
	batches_img,batches_label=random_batch_data(all_data_list_1_2())
	with open("/home/jobs/Desktop/charNumReg/all_data_01.txt","w") as f:
		i=0
		for batch_img in batches_img:
			#f.write(str(batch_img)+"\n")
			print(batches_label[i][19])
			f.write(str(np.argmax(batches_label[i],1)))
			f.write("\n")
			i+=1
