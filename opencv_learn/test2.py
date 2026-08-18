##阈值与图像处理
import cv2
import matplotlib.pyplot as plt
import numpy as np
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#==========图像阈值
# ret,dst = cv2.threshold(src, thresh, maxval, type)
# src:输入图像  thresh:阈值  maxval:当像素超过(小于)了阈值,所赋予的值  type:二值化操作的类型,包含五种
# cv2.THRESH_BINARY:当像素大于阈值时,所赋予的值为maxval,否则为0
# cv2.THRESH_BINARY_INV:当像素大于阈值时,所赋予的为0,否则为maxval
# cv2.THRESH_TRUNC:当像素大于阈值时,所赋予的为阈值,否则为像素本身
# cv2.THRESH_TOZERO:当像素大于阈值时,所赋予的为像素本身,否则为0
# cv2.THRESH_TOZERO_INV:当像素大于阈值时,所赋予的为0,否则为像素本身
img = cv2.imread('cat.jpg')
img_gray = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE)
# cv_show('img',img_gray)    
ret, thresh1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO_INV)
titile = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in range(6):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titile[i])
    plt.xticks([]), plt.yticks([])
# plt.show()

#==========图像平滑处理
img = cv2.imread('lena_noice.png')
# cv_show('img',img)
#均值滤波
# 简单的平均卷积操作
blur = cv2.blur(img,(3,3))
# cv_show('blur',blur)
# 方框滤波
# 基本和均值滤波一样,可以选择归一化,容易越界
box = cv2.boxFilter(img,-1,(3,3),normalize=True)
# cv_show('box',box)
# 高斯滤波
# 高斯模糊的卷积核里的数值是满足高斯分布,相当于更重视中间值
gaussian = cv2.GaussianBlur(img,(5,5),0)
# cv_show('gaussian',gaussian)
# 中值滤波
# 将图像像素点排序,取中间值
median = cv2.medianBlur(img,5)
# cv_show('median',median)
# 展示所有
res = np.hstack((blur,box,gaussian,median))
# cv_show('res',res)

##形态学操作
# ========形态学-腐蚀操作
img = cv2.imread('321.png')
# cv_show('img',img) 
kernel = np.ones((5,5),np.uint8)
erosion_1 = cv2.erode(img,kernel,iterations = 1)  
erosion_2 = cv2.erode(img,kernel,iterations = 3)  
# cv_show('erosion',erosion_1)
# cv_show('erosion',erosion_2)
# ========形态学-膨胀操作
dilate = cv2.dilate(erosion_2,kernel,iterations = 2)
# cv_show('dilate',dilate)

# ========开操作
# 先腐蚀再膨胀
open = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# cv_show('open',open)
# ========闭操作
# 先膨胀再腐蚀
close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
# cv_show('close',close)
# ========形态学-梯度操作
# 梯度 = 膨胀-腐蚀
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
# cv_show('gradient',gradient)

# ========形态学-礼帽操作
# 礼帽 = 输入图像-开操作
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
# cv_show('tophat',tophat)
# ========形态学-黑帽操作
# 黑帽 = 闭操作-输入图像
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
cv_show('blackhat',blackhat)