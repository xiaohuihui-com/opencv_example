import cv2
import matplotlib.pyplot as plt


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
if __name__ == '__main__':
    '''
        image = cv2.imread('../resource/images/2.png',1)
        第一个参数为图片路径
        第二个参数为 1-彩色 0-灰度
    '''
    imgGray = cv2.imread('../resource/images/2.png',0)
    dst1 = Img_adaptiveThreshold(imgGray,1) # ADAPTIVE_THRESH_GAUSSIAN_C + THRESH_BINARY
    dst2 = Img_adaptiveThreshold(imgGray,2) # ADAPTIVE_THRESH_GAUSSIAN_C + THRESH_BINARY_INV
    dst3 = Img_adaptiveThreshold(imgGray,3) # ADAPTIVE_THRESH_MEAN_C + THRESH_BINARY
    dst4 = Img_adaptiveThreshold(imgGray,4) # ADAPTIVE_THRESH_MEAN_C + THRESH_BINARY_INV
    images = [imgGray, dst1,dst2,dst3,dst4]
    titles = ['imgGray','GAUSSIAN_C_BINARY','GAUSSIAN_C_BINARY_INV','MEAN_C_BINARY','MEAN_C_BINARY_INV',
    ]
    plt.figure('adaptiveThreshold deal')
    for i in range(5):
        plt.subplot(2,3,i+1) #行，列，第几个位置
        plt.imshow(images[i],'gray') #灰度显示，默认rgb显示
        plt.title(titles[i]) #每幅图标题
        plt.xticks([]) #横坐标为空
        plt.yticks([]) #纵坐标为空
        # plt.axis('off') #关闭坐标
    plt.show()








