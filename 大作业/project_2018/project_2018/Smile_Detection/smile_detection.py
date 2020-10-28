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

def lbp(imagename):
    image = cv2.imread(imagename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    block=5
    wideth=image.shape[1]
    heigh=image.shape[0]
    column = wideth // block
    row=heigh//block
    lst = []
    for i in range(block*block):
        lbp1 = local_binary_pattern(image[row*(i//block):row*((i//block)+1),column*(i % block):column*((i % block)+1)], 8, 1,  'default')
        hist, _ = np.histogram(lbp1, density=True, bins=256, range=(0, 256))
        lst.append(hist)
    return np.concatenate(lst)

def train_and_test_and_score(train_label,train_histogram,test_label,test_histogram):
    global rptFile
    print("\n\n-----merge finished-----\n\n")
    svc = SVC(kernel='linear', degree=2, gamma=1, coef0=0, C = 2)
    svc.fit(train_histogram,train_label.ravel())
    predict_result=svc.predict(test_histogram)
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
    print("F1: %.6lf"%(2*TP/(2*TP+FP+FN)))
    print("TP: %d ; TN: %d ; FP: %d ; FN: %d ;"%(TP, TN, FP, FN))
    _rpt = open(rptFile, "a")
    _rpt.write("F1: %.6lf\n"%(2*TP/(2*TP+FP+FN)))
    _rpt.write("TP: %d ; TN: %d ; FP: %d ; FN: %d ;\n"%(TP, TN, FP, FN))
    _rpt.close()

def merge(lst):
    Len = len(lst)
    if Len == 1:
        return lst[0]
    M = Len >> 1
    return np.vstack((merge(lst[:M]), merge(lst[M:])))

if __name__ == '__main__':
    rptFile = "report.txt"
    incatalog = "train_and_test"
    _rpt = open(rptFile, "w")
    _rpt.close()
    for time in range(10):
        trainLabel = []
        trainHist = []
        testLabel = []
        testHist = []
        for number in range(0, 10):
            if number == time:
                continue
            print("Testtime: %d  Traingroup: %d  DATA COLLECTION START" % (time, number))
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
            print("Testtime: %d  Traingroup: %d  DATA COLLECTION FINISHED" % (time, number))
        testFaces = open("%s/%d/faces.txt"%(incatalog, time), "r")
        print("Testgroup %d : TEST DATA COLLECTION START"%(time))
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
        print("Testgroup %d : TEST DATA COLLECTION FINISHED"%(time))
        _rpt = open(rptFile, "a")
        _rpt.write("Test group %d:\n"%time)
        _rpt.close()
        train_and_test_and_score(merge(trainLabel), merge(trainHist), merge(testLabel), merge(testHist))