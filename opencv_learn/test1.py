##图像的基本操作
import cv2
import matplotlib.pyplot as plt
import numpy as np
#==========数据的读取-图像 
img = cv2.imread('cat.jpg')
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# cv2.imshow("image",img)
# cv2.waitKey(10000)
# cv2.destroyAllWindows()

def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# print(img.shape)

# img = cv2.imread('cat.png',cv2.IMREAD_GRAYSCALE)
# print(img.shape)

#========数据读取-视频
# vc = cv2.VideoCapture("test.mp4")
# #检测视频是否打开
# if vc.isOpened():
#     open,frame = vc.read()
# else:
#     open = False
# while open:
#     ret,frame = vc.read()
#     if frame is None:
#         break
#     if ret == True:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         cv2.imshow("result",gray)
#         if cv2.waitKey(100) & 0xFF == 27:
#             break
#     vc.release()
#     cv2.destroyAllWindows()

#=========== 截取部分图像数据
# img = img[100:200,100:200]
# cv2.imshow("img",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#========== 颜色通道提取
# b,g,r = cv2.split(img)
# print(b)
# print(g)
# print(r)
# print(r.shape)
# # 只保留R
# img = cv2.merge([r,g,b])
# cur_img = img.copy()
# cur_img[:,:,0] = 0
# cur_img[:,:,1] = 0
# cv2.imshow('R',cur_img)
# cv2.waitKey(0)

#========== 边界填充
# BORDER_REPLICATE: 复制法,复制最边缘的像素
# BORDER_REFLECT: 反射法, abcde|edcba
# BORDER_REFLECT_101: 边缘反射法,edcba|bcde
# BORDER_WRAP: 外包装法,abcde|abcde
# BORDER_CONSTANT: 定值法,用定值填充
# img = cv2.imread('cat.jpg')
# top_size,bottom_size,left_size,right_size = (50,50,50,50)

# replicate = cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,cv2.BORDER_REPLICATE)
# reflect = cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,cv2.BORDER_REFLECT)
# reflect101 = cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,cv2.BORDER_REFLECT_101)
# wrap = cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,cv2.BORDER_WRAP)
# constant = cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,cv2.BORDER_CONSTANT,value=0)
# plt.subplot(231),plt.imshow(img,'gray'),plt.title('ORIGINAL')
# plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
# plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
# plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
# plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
# plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')
# plt.show()

#========== 数值计算 
img_cat = cv2.imread('cat.jpg')
img_dog = cv2.imread('dog.jpg')
# img_cat2 = img_cat+10
# print(img_cat[:5,:5,0])
# print(img_cat2[:5,:5,0])
# print((img_cat+img_dog)[:5,:5,0])

#========== 图像融合
img_dog = cv2.resize(img_dog,(img_cat.shape[1],img_cat.shape[0]))
res = cv2.addWeighted(img_cat,0.5,img_dog,0.5,0)
cv_show('res',res)