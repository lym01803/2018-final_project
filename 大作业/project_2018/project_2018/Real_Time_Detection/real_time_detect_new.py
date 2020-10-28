import cv2
import numpy as np
import os

def addHint(img):
    global _eye, _smile, _rev, _ratio, _recording
    cv2.rectangle(img, (225, 150), (375, 300), (255, 0, 0), 1)
    cv2.putText(img, 'Best Detecting Area', (215, 320), 4, 0.5, (0, 215, 255), 1, cv2.LINE_AA)
    cv2.putText(img, 'Press H To Get Help ', (20, 20), 4, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(img, 'The Input Is Expected To Be Capital Letter', (20, 40), 4, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    if _recording:
        cv2.putText(img, 'Recording', (535, 470), 3, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
    # cv2.putText(img, 'Press Enter To Quit',(390, 20), 4, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    if _eye:
        str = 'Eyes Detection: ON'
    else:
        str = 'Eyes Detection: OFF'
    if _smile:
        str2 = 'Smile Detection: ON'
    else:
        str2 = 'Smile Detection: OFF'
    if _rev:
        cv2.putText(img, 'Reserved', (30, 470), 3, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(img, str, (280, 440), 3, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(img, str2, (280, 470), 3, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(img, 'The Ratio: %.2lf' % _ratio, (30, 440), 3, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(img, 'EXIT', (580, 25), 3, 0.8, (0, 255, 255), 1, cv2.LINE_AA)

def detect(img):
    global _eye, _smile
    #先进行灰度转换
    if img.ndim >= 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 检测人脸：1.利用opencv自带的分类器haarcascade_frontalface_default.xml
    #         2.利用我们自己建立的分类器face_detect.xml(在网上找代码，利用cmd实现）
    facePath = "database/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(facePath)

    # 检测笑脸:1.利用opencv自带的分类器haarcascade_smile.xml
    #         2.利用我们建立的分类器smile_detect.xml
    smilePath = "database/haarcascade_smile.xml"
    smileCascade = cv2.CascadeClassifier(smilePath)
    eyesPath = "database/haarcascade_eye.xml"
    eyesCascade = cv2.CascadeClassifier(eyesPath)
    #检测人脸
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(55, 55),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # 画出每一个人脸，提取出人脸所在区域
    _w, _h = 0, 0
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        if w > _w:
            _w = w
        if h > _h:
            _h = h
        # 对人脸进行笑脸检测
        if _smile:
            smile = smileCascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.16,
                minNeighbors=35,
                minSize=(int(_w / 5), int(_h / 5)),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            # 框出上扬的嘴角并对笑脸打上Smile标签
            for (x2, y2, w2, h2) in smile:
                cv2.rectangle(roi_color, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 185), 1)
                cv2.putText(img, 'Smiling', (x, y - 7), 3, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        if _eye:
            eyes = eyesCascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.16,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            for (x3, y3, w3, h3) in eyes:
                cv2.rectangle(roi_color, (x3, y3), (x3 + w3, y3 + h3), (150, 150, 150), 1)
    return img

cv2.namedWindow('Real-Time Smile Detect', 0)
cv2.resizeWindow('Real-Time Smile Detect', 640, 480)
#cv2.moveWindow('Real-Time Smile Detect', 300, 100)
cv2.imshow('Real-Time Smile Detect', cv2.imread('image/lym_handsome.jpg'))
cv2.waitKey(1)
cap=cv2.VideoCapture(0)

#检测摄像头抓捕照片的宽度以及高度
width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)+0.5)
height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)+0.5)

#视频编码方式
fourcc=cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter('output.avi',fourcc,20.0,(width,height))

#以下是主程序

#global variable
_eye = 0
_smile = 1
_rev = 1
_ratio = 1.0
_recording = 0

#using as constant variable
EYE = ord('E')
SMILE = ord('S')
REV = ord('R')
HELP = ord('H')
LARGER = 44
SMALLER = 46
BACKTOONE = 47
RECORDING = 32

filelist = open("image/list.txt", "r")
while True:
	line = filelist.readline()
	if len(line) == 0 or len(line) == 1:
		break;
	content = line[:-1]
	cv2.imshow('Real-Time Smile Detect', cv2.imread('image/%s'%content))
	cv2.waitKey(50)
#cv2.destroyAllWindows()
_ext = 0
def _exit(event, x, y, flags, param):
    global _ext
    if event == cv2.EVENT_LBUTTONDOWN:
        if x >= 575 and y <= 30:
            _ext = 1
    #print(x,y)
def dealKeyMsg(p):
    global _eye, _smile, _rev, _ratio, _recording
    if p == EYE:
        _eye ^= 1
    elif p == SMILE:
        _smile ^= 1
    elif p == REV:
        _rev ^= 1
    elif p == HELP:
        cv2.namedWindow('help', 0)
        cv2.resizeWindow('help', 640, 480)
        cv2.imshow('help', cv2.imread('image/Help.png'))
    elif p == LARGER:
        if _ratio < 1.5:
            _ratio += 0.01
    elif p == SMALLER:
        if _ratio > 0.5:
            _ratio -= 0.01
    elif p == BACKTOONE:
        _ratio = 1.0
    elif p == RECORDING:
        _recording ^= 1
    return

def Ratio(img):
    global _ratio, _rev
    if _rev:
        img = cv2.flip(frame, 1)
    if np.fabs(_ratio - 1.0) < 1e-6:
        pass
    else:
        tmp = img.shape
        row, col = tmp[:2]
        p1 = np.float32([[col / 2, row / 2], [col / 2 + 1, row / 2], [col / 2, row / 2 + 1]])
        p2 = np.float32([[col / 2, row / 2], [col / 2 + _ratio, row / 2], [col / 2, row / 2 + _ratio]])
        M = cv2.getAffineTransform(p1, p2)
        img = cv2.warpAffine(img, M, (col, row))
    return img

while 1:
    ret,frame=cap.read()
    if ret==True:
        frame = Ratio(frame)
        frame = detect(frame)
        if _recording:
            out.write(frame)
        cv2.namedWindow('Real-Time Smile Detect', 0)
        #cv2.moveWindow('Real-Time Smile Detect', 300, 100)
        cv2.setMouseCallback('Real-Time Smile Detect', _exit)
        addHint(frame)
        cv2.imshow('Real-Time Smile Detect', frame)
        dealKeyMsg(cv2.waitKey(1))
        if _ext:
            break
out.release()
cap.release()
cv2.destroyAllWindows()
#os.system("del *.avi")