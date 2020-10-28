import os
import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QSlider
class winDow(QtWidgets.QWidget):
    '''
    self.title               窗口标题
    self.device              调用设备：内置摄像头
    self.gird                Qt 网格(栅栏)布局
    self.slider              滑动条控件
    self.status_str          label中显示表状态的字符串
    self.press               鼠标按键状态，1为按下，0为释放
    self.delta_x             鼠标拖动位移横坐标
    self.delta_y             鼠标拖动位移纵坐标
    self.ox                  鼠标拖动起始横坐标
    self.oy                  鼠标拖动起始纵坐标
    '''
    def __init__(self):
        super().__init__()
        self.title = "My_Camera"
        self.device = cv2.VideoCapture(0)
        self.grid = QtWidgets.QGridLayout()
        self.setLayout(self.grid)
        self.slider = QSlider(QtCore.Qt.Horizontal, self)
        self.initUI()
        self.status_str = ["off", "on"]
        self.showlb()
        self.press = 0
        self.delta_x = 0
        self.delta_y = 0
        self.ox = 0
        self.oy = 0
    '''
    重写窗口关闭事件
    1.修改CLOSE变量，以便终止进程
    2.弹出对话框，询问用户是否保存录像，并修改DEL变量
    '''
    def closeEvent(self, eV):
        global CLOSE,DEL
        msg = QtWidgets.QMessageBox.information(self, "MSG", "Save the video?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if msg == QtWidgets.QMessageBox.No:
            DEL = 1
        CLOSE = 1
        self.destroy()
    '''
    初始化图像界面
    设置按钮名称，位置，以及设置按键事件
    初始化滑动条
    '''
    def initUI(self):
        self.setWindowTitle(self.title)
        self.resize(800, 600)
        self.label = QtWidgets.QLabel()
        self.grid.addWidget(self.label, 0, 0, 6, 6)
        pos = [(6, 0), (6, 1), (6, 2), (6, 3), (6, 4)]
        pos2 = [(1, 6), (2, 6), (3, 6), (4, 6), (5, 6)]
        name = ["Reverse", "Smile", "Eye", "Reset", "Record"]
        for _p, _n  in zip(pos, name):
            button = QtWidgets.QPushButton(_n)
            self.grid.addWidget(button, *_p)
            button.clicked.connect(self.buttonClicked)
        self.lb = []
        for _p in pos2:
            tmplb = QtWidgets.QLabel()
            tmplb.setFont(QtGui.QFont("Consolas", 12, QtGui.QFont.Bold))
            self.grid.addWidget(tmplb, *_p)
            self.lb.append(tmplb)
        self.grid.addWidget(self.slider, *(6, 6))
        self.slider.setRange(0, 100)
        self.slider.valueChanged.connect(self.on_change)
    '''
    处理滑动条valueChange事件
    '''
    def on_change(self):
        global _ratio
        _ratio = self.slider.value() * 0.01 + 0.5
        self.showlb()
        return
    '''
    在状态发生改变时，显示最新的label信息
    '''
    def showlb(self):
        global _smile, _eye, _rev, _ratio, _recording
        self.lb[0].setText("  smile detection: %s"%self.status_str[_smile])
        self.lb[1].setText("  eye detection  : %s"%self.status_str[_eye])
        self.lb[2].setText("  reverse        : %s"%self.status_str[_rev])
        self.lb[3].setText("  record         : %s"%self.status_str[_recording])
        self.lb[4].setText("  ratio:         : %.2lf"%_ratio)
        self.slider.setValue(int((_ratio - 0.5) * 100))
    '''
    按键事件
    "+","-"为旧版本的功能，现已由滑动条替代
    '''
    def buttonClicked(self):
        global _eye, _smile, _rev, _ratio, _recording
        txt = self.sender().text()
        if txt == "Reverse":
            _rev ^= 1
        elif txt == "Smile":
            _smile ^= 1
        elif txt == "Eye":
            _eye ^= 1
        elif txt == "Reset":
            _ratio = 1.0
        elif txt == "Record":
            _recording ^= 1
        self.showlb()
        #调用self.showlb()实时显示最新状态
    '''
    加工处理并显示摄像头返回的图像
    '''
    def showImage(self):
        global _recording, _out
        ret, frame = self.device.read()
        if not ret:
            return
        cv2.waitKey(0)
        if self.delta_x != 0 or self.delta_y != 0:
            frame = self.Move(frame)    #平移
        frame = Ratio(frame)            #翻转和缩放
        frame = detect(frame)           #探测脸部和微笑
        if _recording:
            _out.write(frame)           #如果record处于on状态，就向视频文件中写入当前图片
        self.showimg(frame)
    '''
    加载过场动画
    '''
    def loadingPicture(self):
        global CLOSE
        self.showimg(cv2.imread('image/lym_handsome.jpg'))
        cv2.waitKey(1500)
        filelist = open("image/list.txt", "r")
        while True:
            line = filelist.readline()
            if len(line) == 0 or len(line) == 1 or CLOSE:
                break
            content = line[:-1]
            self.showimg(cv2.imread('image/%s' % content))
            cv2.waitKey(50)
    '''
    将cv2返回的图片，转化为Qlabel能够方便显示的像素图
    '''
    def showimg(self, img):
        _h, _w, bPC = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.image = QtGui.QImage(img.data, _w, _h, bPC * _w, QtGui.QImage.Format_RGB888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(self.image))
    '''
    判定鼠标位置是否在指定区域
    '''
    def in_area(self, x, y):
        return 20 <= x and x <= 640 and 45 <= y and y <= 510    #判定鼠标是否位于图像所在的label区域
    '''
    处理鼠标按下事件
    '''
    def mousePressEvent(self, event):
        x = event.x()
        y = event.y()
        if self.in_area(x, y):
            #print(event.x(), event.y())
            self.press = 1              #记录鼠标按键状态为按下
            self.ox = x
            self.oy = y
        return
    '''
    处理鼠标释放事件
    '''
    def mouseReleaseEvent(self, event):
        #print("released")
        self.press = 0          #记录鼠标按键状态为释放
        self.delta_x = 0        #清零位移
        self.delta_y = 0
    '''
    处理鼠标移动事件
    和上面两个函数一起，实现鼠标拖动画面的功能
    '''
    def mouseMoveEvent(self, event):
        x = event.x()
        y = event.y()
        if self.press:
            self.delta_x = x - self.ox
            self.delta_y = y - self.oy
            #print("delta : %d %d %d %d %d %d"%(self.ox, self.oy, self.delta_x, self.delta_y, x, y))
            if self.delta_x > 160:
                self.delta_x = 160
            if self.delta_x < -160:
                self.delta_x = -160
            if self.delta_y > 120:
                self.delta_y = 120
            if self.delta_y < -120:
                self.delta_y = -120    #防止平移的时候越界导致错误
        return
    '''
    实现图像的平移
    '''
    def Move(self, img):
        global rev
        x = self.delta_x
        y = self.delta_y
        if _rev:
            x = -x
        tmp = img.shape
        row, col = tmp[:2]
        p1 = np.float32([[col/2, row/2], [col/2 + 1, row/2], [col/2, row/2 + 1]])
        p2 = np.float32([[col/2 + x, row/2 + y], [col/2 + 1 + x, row/2 + y], [col/2 + x, row/2 + 1 + y]])
        M = cv2.getAffineTransform(p1, p2)   #生成转换矩阵; 要求 p1, p2 中的三个点不共线
        img = cv2.warpAffine(img, M, (col, row))
        return img

'''
重要的全局变量，表示程序的各个功能开启状态和一些重要的参数
'''
#global variable
_eye = 0            #是否开启眼部识别 0:off, 1:on, 默认关闭
_smile = 1          #是否开启笑脸识别 0:0ff, 1:on, 默认开启
_rev = 1            #是否开启翻转     0:off, 1:on, 默认开启
_ratio = 1.0        #当前缩放比例     range: 0.5~1.5 默认为 1.0
_recording = 0      #是否开启录像     0:off, 1:on, 默认关闭
CLOSE = 0
DEL = 0             #是否保存视频文件  0:删除  1:保留

'''
实现对脸部，微笑，眼睛的探测
'''
def detect(img):
    global _eye, _smile
    if img.ndim >= 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    facePath = "database/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(facePath)
    smilePath = "database/haarcascade_smile.xml"
    smileCascade = cv2.CascadeClassifier(smilePath)
    eyesPath = "database/haarcascade_eye.xml"
    eyesCascade = cv2.CascadeClassifier(eyesPath)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(55, 55),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    _w, _h = 0, 0
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        if w > _w:
            _w = w
        if h > _h:
            _h = h
        #print(_w,_h)
        if _smile:
            smile = smileCascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.16,
                minNeighbors=35,
                minSize=(max(5, int(0.4*_w - 30 + 0.5)), max(5, int(0.4*_h - 30 + 0.5))),     #玄学调参，请勿在意
                flags=cv2.CASCADE_SCALE_IMAGE
            )
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
'''
实现图像的翻转和缩放
'''
def Ratio(img):
    global _ratio, _rev
    if _rev:
        img = cv2.flip(img, 1)
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

'''
主函数部分
'''
if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    wd = winDow()

    width = int(wd.device.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(wd.device.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    _out = cv2.VideoWriter('output.avi', fourcc, 12.0, (width, height))     #经试验 .avi格式不会报错，不会 warning, 具体原因不详

    wd.show()
    wd.loadingPicture()   #作者丑照显示 & 过场动画加载
    while True:
        wd.showImage()
        if CLOSE:
            break

    wd.device.release()
    _out.release()
    if DEL:
        os.system("del output.avi")

    exit()