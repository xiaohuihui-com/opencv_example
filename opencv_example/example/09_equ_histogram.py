import cv2
import matplotlib.pyplot as plt
import numpy as np

''' 打印图片信息 '''
def getImgInfo(image):
    print("类型：{}".format(type(image)))
    print("图片尺寸：{}".format(image.shape))
    print("图片大小：{}".format(image.size))
    print("图片存储类型：{}".format(image.dtype))

''' 图片显示 '''
def showImg(image):
    cv2.imshow("show",image)
    cv2.moveWindow("show",400,40)# 设置窗口位置
    cv2.resizeWindow("show", image.shape[1], image.shape[0]);
    cv2.waitKey(0) #等待时间，毫秒级，0表示任意键终止
    cv2.destroyAllWindows()

''' 等比例缩放图片 '''
def resizeImg(image,ratio):
    weight = image.shape[0]
    height = image.shape[1]
    img = cv2.resize(image,(int(weight*ratio),int(height*ratio)))
    return img

''' 二值化阈值处理 '''
def Img_threshold(imgGray, mode, thresh=127, maxval=255):
    if mode == 1:                                   #二值化
        ret ,dst =cv2.threshold(imgGray,thresh,maxval,cv2.THRESH_BINARY)
    elif mode == 2:                                 #反向二值化
        ret ,dst =cv2.threshold(imgGray,thresh,maxval,cv2.THRESH_BINARY_INV)
    elif mode == 3:                                 #截断二值化
        ret ,dst =cv2.threshold(imgGray,thresh,maxval,cv2.THRESH_TRUNC)
    elif mode == 4:                                 #低阈值零处理二值化
        ret ,dst =cv2.threshold(imgGray,thresh,maxval,cv2.THRESH_TOZERO)
    elif mode == 5:                                 #超阈值零处理二值化
        ret ,dst =cv2.threshold(imgGray,thresh,maxval,cv2.THRESH_TOZERO_INV)
    return dst


''' 二值化阈值处理 '''
def Img_threshold(imgGray, mode, thresh=127, maxval=255):
    if mode == 1:                                   #二值化
        ret ,dst =cv2.threshold(imgGray,thresh,maxval,cv2.THRESH_BINARY)
    elif mode == 2:                                 #反向二值化
        ret ,dst =cv2.threshold(imgGray,thresh,maxval,cv2.THRESH_BINARY_INV)
    elif mode == 3:                                 #截断二值化
        ret ,dst =cv2.threshold(imgGray,thresh,maxval,cv2.THRESH_TRUNC)
    elif mode == 4:                                 #低阈值零处理二值化
        ret ,dst =cv2.threshold(imgGray,thresh,maxval,cv2.THRESH_TOZERO)
    elif mode == 5:                                 #超阈值零处理二值化
        ret ,dst =cv2.threshold(imgGray,thresh,maxval,cv2.THRESH_TOZERO_INV)
    return ret, dst

''' 自适应阈值处理 '''
def Img_adaptiveThreshold(imgGray,mode,size=5):
    if mode == 1:
        athd = cv2.adaptiveThreshold(imgGray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,size,3)
    elif mode == 2:
        athd = cv2.adaptiveThreshold(imgGray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,size,3)
    elif mode == 3:
        athd = cv2.adaptiveThreshold(imgGray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,size,3)
    elif mode == 4:
        athd = cv2.adaptiveThreshold(imgGray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,size,3)
    return athd

''' 大率法 '''
def Img_otsu(imgGray,mode):
    if mode == 1:
        t,otsu = cv2.threshold(imgGray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    elif mode == 2:
        t,otsu = cv2.threshold(imgGray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return otsu

''' 滤波处理 '''
def Img_filter(imgGray,mode):
    if mode == 1:       # 均值滤波
        dst = cv2.blur(imgGray,(5,5))
    elif mode == 2:     # 方框滤波
        dst = cv2.boxFilter(imgGray,-1,(5,5))
    elif mode == 3:     # 高斯滤波
        dst = cv2.GaussianBlur(imgGray,(5,5),0,0)
    elif mode == 4:     # 中值滤波
        dst = cv2.medianBlur(imgGray,3)
    elif mode == 5:     # 双边滤波
        dst = cv2.bilateralFilter(imgGray,25,100,100)
    elif mode == 6:     # 2D卷积
        dst = cv2.filter2D(imgGray,-1,2)
    return dst

''' 形态学处理 '''
def Img_morphology(imgGray,mode,times=1):
    kernel = np.ones((5,5),np.uint8)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))    #矩形核
    kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))   #十字核
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)) #椭圆核
    if mode == 1:   	  #腐蚀
        dst = cv2.erode(imgGray,kernel,iterations=times)
    elif mode == 2: 	  #膨胀
        dst = cv2.dilate(imgGray,kernel,iterations=times)
    elif mode == 3:       #开运算
        dst = cv2.morphologyEx(imgGray,cv2.MORPH_OPEN,kernel,iterations=times)
    elif mode == 4:       #闭运算
        dst = cv2.morphologyEx(imgGray,cv2.MORPH_CLOSE,kernel,iterations=times)
    elif mode == 5:       #形态学梯度运算
        dst = cv2.morphologyEx(imgGray,cv2.MORPH_GRADIENT,kernel,iterations=times)
    elif mode == 6:       #礼帽运算
        dst = cv2.morphologyEx(imgGray,cv2.MORPH_TOPHAT,kernel,iterations=times)
    elif mode == 7:		  #黑帽运算
        dst = cv2.morphologyEx(imgGray,cv2.MORPH_BLACKHAT,kernel,iterations=times)
    return dst

''' Soble边缘检测 '''
def Img_Soble(imgGray,alpha=0.5,beta=0.5,gamma=10):
    dstx = cv2.Sobel(imgGray,cv2.CV_64F,1,0)#对x
    dsty = cv2.Sobel(imgGray,cv2.CV_64F,0,1)#对y
    dstx = cv2.convertScaleAbs(dstx)   #对运算结果取绝对值
    dsty = cv2.convertScaleAbs(dsty)   #对运算结果取绝对值
    dstxy = cv2.addWeighted(dstx,alpha,dsty,beta,gamma)
    return dstxy

''' Scharr边缘检测 '''
def Img_Scharr(imgGray,alpha=0.5,beta=0.5,gamma=10):
    dstx = cv2.Scharr(imgGray,cv2.CV_64F,1,0)#对y
    dsty = cv2.Scharr(imgGray,cv2.CV_64F,0,1)#对y
    dstx = cv2.convertScaleAbs(dstx)   #对运算结果取绝对值
    dsty = cv2.convertScaleAbs(dsty)   #对运算结果取绝对值
    dstxy = cv2.addWeighted(dstx,alpha,dsty,beta,gamma)
    return dstxy

''' Canny边缘检测 '''
def Img_Canny(imgGray,threshold1=128,threshold2=255):
    dst = cv2.Canny(imgGray,threshold1,threshold2)
    return dst
if __name__ == '__main__':
    '''
        image = cv2.imread('../resource/images/2.png',1)
        第一个参数为图片路径
        第二个参数为 1-彩色 0-灰度
    '''
    imgGray = cv2.imread('../resource/images/2.png',0)
    equ = cv2.equalizeHist(imgGray) # 均衡直方图
    images = [imgGray, equ]
    titles = ['imgGray','equ']
    plt.figure('morphology deal')
    plt.subplot(2,2,1)
    plt.hist(imgGray.ravel())
    plt.subplot(2,2,2)
    plt.imshow(imgGray,'gray')
    plt.subplot(2,2,3)
    plt.hist(equ.ravel())
    plt.subplot(2,2,4)
    plt.imshow(equ,'gray')

    plt.show()








