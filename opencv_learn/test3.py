##图像梯度-Sobel算子
import cv2
import matplotlib.pyplot as plt
import numpy as np
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#==========sobel算子
# dst = cv2.Sobel(src,ddepth,dx,dy,ksize)
# src:输入图像  ddepth:输出图像的深度  dx:x方向的导数  dy:y方向的导数  ksize:sobel核的大小
img = cv2.imread('dog.jpg', cv2.IMREAD_GRAYSCALE)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
# cv_show('sobelx',sobelx)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
sobely = cv2.convertScaleAbs(sobely)
# cv_show('sobely',sobely)
sobelxy = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
# cv_show('sobelxy',sobelxy)

# ===========Scharr算子
scharrx = cv2.Scharr(img,cv2.CV_64F,1,0)
scharry = cv2.Scharr(img,cv2.CV_64F,0,1)
scharrxy = cv2.addWeighted(scharrx,0.5,scharry,0.5,0)
# cv_show('scharrxy',scharrxy)

# ===========Laplacian算子
laplacian = cv2.Laplacian(img,cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)
# cv_show('laplacian',laplacian)
res = np.hstack((sobelxy,scharrxy,laplacian))
# cv_show('res',res)


##Canny边缘检测
# 1）使用高斯滤波器，以平滑图像滤除噪声
# 2）计算图像中每个像素点的梯度强度和方向
# 3）使用非极大值抑制，去除边缘检测带来的杂散响应
# 4）使用双阈值算法，确定边缘点
# 5）通过抑制孤立的弱边缘最终完成边缘检测
v1 = cv2.Canny(img,30,70)
# cv_show('img',img)
# cv_show('v1',v1)

# ===========图像轮廓
# cv2.findContours(src,mode,method)
# mode: 轮廓的检索模式  method: 轮廓的近似方法
# mode参数: cv2.RETR_EXTERNAL:只检测外轮廓 
# cv2.RETR_LIST:检测所有轮廓,并将其保存到一条链表当中
# cv2.RETR_CCOMP:检测所有的轮廓，并将他们组织为两层：顶层是各部分的外部边界，第二层是空洞的边界  
# cv2.RETR_TREE:检测所有轮廓，并重构嵌套轮廓的整个层次
# method参数: cv2.CHAIN_APPROX_NONE:存储所有的轮廓点  
# cv2.CHAIN_APPROX_SIMPLE:压缩水平、垂直和斜线段，只保留他们的终点坐标
# 为了更加准确，使用二值图像
img = cv2.imread('dog.jpg')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
# cv_show('thresh',thresh)
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
# 绘制轮廓
draw_img = img.copy()
# 传入绘制图像，轮廓，轮廓索引，颜色模式，线条厚度
# 注意需要copy,要不图会变
res = cv2.drawContours(draw_img,contours,-1,(0,0,255),2)
cv_show('res',res)
# 获取轮廓特征
cnt = contours[0]
# 面积
area = cv2.contourArea(cnt)
# 周长,True表示闭合的
perimeter = cv2.arcLength(cnt,True)

# ================模板匹配
# TM_SQDIFF：计算平方不同，计算出来的值越小，越相关
# TM_CCORR:计算相关性，计算出来的值越大，越相关
# TM_CCOEFF:计算相关系数，计算出来的值越大，越相关
# TM_SQDIFF_NORMED：计算归一化平方不同，计算出来的值越接近0，越相关
# TM_CCORR_NORMED：计算归一化相关性，计算出来的值越接近1，越相关
# TM_CCOEFF_NORMED：计算归一化相关系数，计算出来的值越接近1，越相关
template = cv2.imread('dog_face.jpg')
template_gray = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
res = cv2.matchTemplate(img_gray,template_gray,1)
h,w = template.shape[:2]
methods = ["cv2.TM_SQDIFF","cv2.TM_CCORR","cv2.TM_CCOEFF","cv2.TM_SQDIFF_NORMED","cv2.TM_CCORR_NORMED","cv2.TM_CCOEFF_NORMED"]
min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)
# print(min_val,max_val,min_loc,max_loc)
for meth in methods:
    img2 = img.copy()
    # 匹配方法的真值
    method = eval(meth)
    res = cv2.matchTemplate(img,template,method)
    min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)
    # 如果是平方差匹配TM_SQDIFF或归一化平方差匹配TM_SQDIFF_NORMED，取最小值
    if method in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0]+w,top_left[1]+h) 
    #绘制矩形
    cv2.rectangle(img2,top_left,bottom_right,255,2)
    plt.subplot(121),plt.imshow(res,cmap='gray')
    # plt.xticks([]),plt.ytics([])隐藏坐标轴
    plt.subplot(122),plt.imshow(img2,cmap='gray')
    plt.xticks([]),plt.yticks([])
    plt.suptitle(meth)
    # plt.show()

#============匹配多个对象
img_rgb = cv2.imread('number.jpg')
img_gray = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)
template = cv2.imread('number_4.jpg',0)
h,w = template.shape[:2]

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.8
# 去匹配度大于80%的图像
loc = np.where(res>=threshold)
for pt in zip(*loc[::-1]): #*表示可选参数
    bottom_right = (pt[0]+w,pt[1]+h)
    cv2.rectangle(img_rgb,pt,bottom_right,(0,0,255),2)
# cv_show('img_rgb',img_rgb)


# ============图像金字塔
# 高斯金字塔：向下采样方法（缩小）
# 将Gi与高斯内核卷积；将所有偶数行和列去除
# 高斯金字塔：向上采样方法（放大）
# 将图像在每个方向扩大为原来的两倍，新增的行和列以0为填充；使用先前同样的内核（乘4）与放大后的图像卷积，获得近似值
img = cv2.imread('dog_face.jpg')
# cv_show('img',img)
# print(img.shape)
up = cv2.pyrUp(img)
# cv_show('up',up)
# print(up.shape)
down = cv2.pyrDown(img)
# cv_show('down',down)
# print(down.shape)
up_down = cv2.pyrDown(up)
# cv_show('up_down',up_down)
# print(up_down.shape)
# 拉普拉斯金字塔
# 1.低通滤波 2.缩小尺寸 3.放大尺寸 4.图像相减
