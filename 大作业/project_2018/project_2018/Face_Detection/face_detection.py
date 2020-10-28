"""
 1.在该py文件根目录下有：
                    ①files_of_faces文件夹：
                                          A.存储提供的所有的照片
                                          B.所有照片的名字（利用cmd指令实现
                    ②the_frame_of_faces文件夹：
                                          存储能框出脸部且框出脸部后的照片
 2.若人脸无法检测出来，则不显示
 3.显示人脸辨别率
"""


import cv2,os
from PIL import Image,ImageDraw



#利用opencv自带分类器haarcascade_frontalface_default.xml来检测人脸
def detectFaces(image_name):
    img = cv2.imread(image_name)
    face_cascade = cv2.CascadeClassifier("trained_files/haarcascade_frontalface_default.xml") # classifier
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    result = []
    for (x,y,width,height) in faces:
        result.append((x,y,x+width,y+height))
    return result


#用红线划出人脸部分
def drawFaces(image_name):
    global count, nondetected, out_catalog
    faces = detectFaces(image_name)
    img = Image.open(image_name)
    draw_instance = ImageDraw.Draw(img)
    if len(faces):                      # to avoid counting more than once when detecting more than one faces in one picture
        count+=1
    else:
        nondetected.append(image_name)  # if faces is empty, this picture must be not detected
    for (x1,y1,x2,y2) in faces:
        draw_instance.rectangle((x1,y1,x2,y2), outline=(255))
        img.save(out_catalog + '/' + image_name.split('/')[1])

"""
    以下是主程序
    变量说明：
            count   识别的人脸数   int
            sum     总人脸数       int
            faces   文件          file
            theline 文件某一行     string
            content 照片地址       string
            
"""
if __name__ == '__main__':
    in_catalog = "files_of_faces"           # use a variable to represent the string, make it more convenient to modify the path
    out_catalog = "the_frame_of_faces_new"
    count=0
    sum=0
    error = []
    nondetected = []
    faces = open(in_catalog + '/faces.txt', 'r')
    if not os.path.exists(out_catalog):

    #若不存在the_frame_of_faces文件，则创建

        os.mkdir(out_catalog)
    else:
        os.system("del %s\\*.jpg"%out_catalog)
    while True:
        theline = faces.readline()
        if len(theline) == 0 or len(theline)==1:
            break
        sum+=1
        print("%d"%sum)
        content = theline[:-1]
        try:                         # Robustness. To avoid losing data that we have already got when some errors occur.
            drawFaces(in_catalog + '/' + content)
        except ValueError:
            error.append(in_catalog + '/' + content)
        except RuntimeError:
            error.append(in_catalog + '/' + content)
    faces.close()
    _rpt = open("FaceDetectionReport.txt", "w");
    print('共接收'+str(sum)+'张照片')
    print('共识别'+str(count)+'张照片')
    print('识别率：'+str(count/sum))
    _rpt.write('received %d photos\n'%(sum))
    _rpt.write('detected %d photos\n'%(count))
    _rpt.write('detecting rate: %.2lf%%\n'%(100*count/sum)) # write the result in the file: FaceDetectionReport.txt
    if len(error):                                          # if there are some errors write in the TXT
        _rpt.write("\nError occured:\n")
        for _e in error:
            _rpt.write("%s\n"%_e)
    if len(nondetected):                                    # if there are photos not detected, write the filename in the TXT
        _rpt.write("\nnot detected:\n")
        for _n in nondetected:
            _rpt.write("%s\n"%_n)
    _rpt.close()
    #os.system("pause")