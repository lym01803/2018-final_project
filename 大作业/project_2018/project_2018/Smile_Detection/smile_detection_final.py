from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data, io
from skimage.color import label2rgb
import skimage
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import matplotlib.image as mpimg
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVR
from skimage import feature as skft
from sklearn.svm import SVC
import os


def lbp(imagename):#输入：一张人脸部分的照片的名字（string），输出：这张照片的特征向量#
	image = cv2.imread(imagename)#image:照片#
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#image:灰度照片#
	imagepart=[]
	block=6
	wideth=image.shape[1]
	heigh=image.shape[0]#读取输入图片的高度与宽度,也即是各有多少像素点数#
	column = wideth // block
	row=heigh//block#计算切割后，每个部分应该有多少行多少列,也即是各有多少像素点数#
	lst = []
	for i in range(block*block):
		lbp1 = local_binary_pattern(image[row*(i//block):row*((i//block)+1),column*(i % block):column*((i % block)+1)], 8, 1,  'default')
		hist, _ = np.histogram(lbp1, density=True, bins=256, range=(0, 256))
		lst.append(hist)
	return np.concatenate(lst)

def train_and_test_and_score(train_label,train_histogram,test_label,test_histogram):
	#输入：test_histogram含有3600个histogram（对应3600张图片的特征向量），test_label含3600个1或-1（对应3600张图片的标签）#
	#输出：得   分#
	print("\n\nTraining...\n\n")
	svc = SVC(kernel='linear', degree=2, gamma=1, coef0=0, C = 0.755784)
	svc.fit(train_histogram,train_label.ravel())#训练#
	print("Training finished...\n\nStart detecting...\n\n")
	predict_result=svc.predict(test_histogram)#测试，predict含有400个1或-1#
	TP = 0
	FP = 0
	FN = 0
	TN = 0
	for i in range(len(predict_result)):
		if test_label[i]==1:
			if predict_result[i]==1:
				TP+=1
			else:
				FN+=1
		else:
			if predict_result[i]==1:
				FP+=1
			else:
				TN+=1
	_rpt = open("report.txt", "a");
	_rpt.write("F1: %.6lf\n"%(2*TP/(2*TP+FP+FN)))
	_rpt.write("TP: %d ; TN: %d ; FP: %d ; FN: %d ;\n\n"%(TP, TN, FP, FN))
	_rpt.close()

def train_and_test2(train_label,train_histogram,test_histogram,file_names):
	print("\n\nTraining...\n\n")
	svc=SVC(kernel="linear",degree=2,gamma=1,coef0=0,C=0.755784)
	svc.fit(train_histogram,train_label.ravel())
	print("Training finished...\n\nStart detecting...\n\n")
	predict_result=svc.predict(test_histogram)
	len_of_result=len(predict_result);
	is_smile=0;not_smile=0
	name_of_smiles=[];name_of_other=[]
	for i in range(len_of_result):
		if predict_result[i]==1:
			is_smile+=1
			name_of_smiles.append(file_names[i])
		else:
			not_smile+=1
			name_of_other.append(file_names[i])
	with open("report.txt","a") as _rp:
		if is_smile>0:
			str1="images are";str2="image is"
			str3="smiling faces";str4="a smiling face"
			_rp.write("The following {} considered as {}:\n".format(str2 if is_smile==1 else str1,str4 if is_smile==1 else str3))
			for Names in name_of_smiles:
				_rp.write(Names+"\n")
			_rp.write("\n\n\n")
		else:
			_rp.write("There is no smiling face!\n\n\n")
		if not_smile>0:
			str1="images are";str2="image is"
			str3="smiling faces";str4="a smiling face"
			_rp.write("The following {} not considered as {}:\n".format(str2 if is_smile==1 else str1,str4 if is_smile==1 else str3))
			for Names in name_of_other:
				_rp.write(Names+"\n")

if __name__ == '__main__':
	incatalog = "train_and_test"
	_rpt = open("report.txt", "w")
	_rpt.close()
	for time in range(10):
		trainLabel = []
		trainHist = []
		testLabel = []
		testHist = []
		for number in range(0, 10):
			if number == time:
				continue
			print("Test_time: %d  Train_group: %d  DATA COLLECTION START" % (time, number))
			trainFaces = open("%s/%d/faces.txt"%(incatalog, number), "r")
			while True:
				line = trainFaces.readline()
				if len(line) == 0 or len(line) == 1:
					break
				content = line[:-1].split()[0]
				label = int(line[:-1].split()[1][0])
				hist = lbp("%s/%d/%s"%(incatalog, number, content))
				trainLabel.append(label)
				trainHist.append(hist)
			trainFaces.close()
			print("Test_time: %d  Train_group: %d  DATA COLLECTION FINISHED" % (time, number))
		testFaces = open("%s/%d/faces.txt"%(incatalog, time), "r")
		print("Test_group %d : TEST DATA COLLECTION START"%(time))
		while True:
			line = testFaces.readline()
			if len(line) == 0 or len(line) == 1:
				break
			content = line[:-1].split()[0]
			label = int(line[:-1].split()[1][0])
			hist = lbp("%s/%d/%s"%(incatalog, time, content))
			testLabel.append(label)
			testHist.append(hist)
		testFaces.close()
		print("Test_group %d : TEST DATA COLLECTION FINISHED"%(time))
		_rpt = open("report.txt", "a")
		_rpt.write("Test_group %d:\n"%time)
		_rpt.close()
		train_and_test_and_score(np.vstack(trainLabel), np.vstack(trainHist), np.vstack(testLabel), np.vstack(testHist))
		print("Test_group %d Finished!!\n\n"%(time))
	# the following part is to detect smiles in the given extra images 
	new_data="own_test_set"
	trainLabel = []
	trainHist = []
	testLabel = []
	testHist = []
	test_his=[]
	train_his=[]
	train_lab=[]
	file_na=[]
	if os.path.exists(new_data):
		# the following part is to collect the names of the images in the folder "own_test_set"
		fa_path=os.path.dirname(os.path.abspath(__file__))
		os.chdir(r"{}".format(fa_path+"/"+new_data))
		os.system("dir /b *.png *.jpg >names.txt")
		os.chdir(fa_path)
		# the following part is to collect all the given images
		with open(new_data+"/names.txt","r") as Na:
			is_empty=True
			while True:
				line=Na.readline()
				if len(line)==0 or len(line)==1:
					break
				is_empty=False
				content=line[:-1].split()[0]
				hist=lbp(new_data+"/"+content)
				test_his.append(hist)
				file_na.append(content)
		if is_empty==True:
			print("There is no extra image to be detected!")
			with open("report.txt","a") as _rp:
				_rp.write("There is no extra image to be detected!")
		else:
			print("Start the training SVM and detecting the extra image{}...\n\n".format("" if len(file_na)==1 else "s"))
			with open("report.txt","a") as _rp:
				_rp.write("\n\n\nThe following part is the test result of the extra image{}:\n".format("" if len(file_na)==1 else "s"))
		# the following part is to collect the train histogram
		for number in range(0,9):
			with open(incatalog+"\\"+str(number)+"\\faces.txt","r") as trainFaces:
				print("Train_group %d DATA COLLECTION START"%(number))
				while True:
					line=trainFaces.readline()
					if len(line)==0 or len(line)==1:
						break
					content=line.split()[0]
					label=int(line.split()[1][0])
					hist=lbp(r"{}/{}/{}".format(incatalog,number,content))
					train_his.append(hist)
					train_lab.append(label)
				print("Train_group %d DATA COLLECTION FINISHED"%(number))
		train_and_test2(np.vstack(train_lab),np.vstack(train_his),np.vstack(test_his),file_na)
		print("the extra group finished!!\n\n".title())
		print('The test result is stored in "{}".'.format(fa_path+"\\"+"report.txt"))
#os.system("pause")