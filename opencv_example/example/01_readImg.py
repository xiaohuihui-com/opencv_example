import cv2


''' 打印图片信息 '''
def getImgInfo(image):
    print("类型：{}".format(type(image)))
    print("图片尺寸：{}".format(image.shape))
    print("图片大小：{}".format(image.size))
    print("图片存储类型：{}".format(image.dtype))

''' 图片显示 '''
def showImg(image):
    cv2.imshow("show",image)
    cv2.moveWindow("show",400,200)# 设置窗口位置
    cv2.resizeWindow("show", image.shape[1], image.shape[0]);
    cv2.waitKey(0) #等待时间，毫秒级，0表示任意键终止
    cv2.destroyAllWindows()

if __name__ == '__main__':
    '''
        image = cv2.imread('../resource/images/2.png',1)
        第一个参数为图片路径
        第二个参数为 1-彩色 0-灰度
    '''
    image = cv2.imread('../resource/images/2.png',1)
    showImg(image)
    getImgInfo(image)