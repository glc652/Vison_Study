import cv2
import numpy as np
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''
    1.图像读取与显示
'''
def show_color_and_gray(image_path):
    img_color = cv2.imread(image_path)

    if img_color is None:
        print("无法读取图片，请检查路径是否正确。")
        return

    # 输出图像的宽、高、通道数
    height, width, channels = img_color.shape
    print(f"图像尺寸: 宽={width}, 高={height}, 通道数={channels}")

    # 转换为灰度图
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # 同时显示彩色和灰度图像
    cv2.imshow("Color Image", img_color)
    cv2.imshow("Gray Image", img_gray)

    # 等待任意按键后关闭所有窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''
    2. 图像保存与格式转换
'''
def load_convert_save_and_show(image_path, output_path):
    img_color = cv2.imread(image_path)

    if img_color is None:
        print("无法读取图片，请检查路径是否正确。")
        return

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    
    cv2.imwrite(output_path, img_gray)
    print(f"灰度图像已保存为 {output_path}")

    # 同时显示原图和灰度图
    cv2.imshow("Color Image", img_color)
    cv2.imshow("Gray Image", img_gray)
    
    # 等待任意按键后关闭所有窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''
    3. 实时打开摄像头并显示视频流
'''
def show_camera_feed():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头，请检查设备连接。")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取视频帧。")
            break

        cv2.imshow("Camera Feed", frame)

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

'''
    4. 从摄像头中捕获单张图片
    示例:capture_from_camera("D:\\vision\\cv\\test1\\pic\\capture.jpg")
'''
def capture_from_camera(save_path="capture.jpg"):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头，请检查设备连接。")
        return

    print("摄像头已启动。按 's' 保存照片，按 'q' 退出。")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取视频帧。")
            break

        cv2.imshow("Camera Capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            cv2.imwrite(save_path, frame)
            print(f"照片已保存为 {save_path}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

'''
    5. 读取视频文件并逐帧播放
    示例:play_video("D:\\vision\\cv\\test1\\pic\\目标检测车辆检测测试视频.mp4")
'''
def play_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    print("正在播放视频... 按 'q' 退出。")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频播放完毕或无法读取帧。")
            break

        cv2.imshow("Video Player", frame)

        # 控制播放速度（约30ms/帧 ≈ 33 FPS）
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

'''
    6. 统计图像基本信息
    示例: analyze_image("D:\\vision\\cv\\test1\\pic\\football.jpg")
'''
def analyze_image(image_path):
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"无法读取图像: {image_path}")
        return

    # 获取高、宽、通道数（如果存在）
    if len(img.shape) == 3:
        height, width, channels = img.shape
        is_color = True
    else:
        height, width = img.shape
        channels = 1
        is_color = False

    total_pixels = height * width
    dtype = img.dtype

    # 转为灰度图（统一计算平均亮度）
    if is_color:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    mean_brightness = np.mean(gray)

    # 输出结果
    print(f"图像路径: {image_path}")
    print(f"宽度: {width}")
    print(f"高度: {height}")
    print(f"数据类型: {dtype}")
    print(f"像素总数: {total_pixels}")
    print(f"平均亮度（灰度）: {mean_brightness:.2f}")

'''
    7. 摄像头实时灰度化显示
'''
def gray_camera():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头，请检查设备连接。")
        return

    print("正在启动灰度摄像头... 按 'q' 退出。")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取视频帧。")
            break

        # 转为灰度图
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 显示灰度图像，窗口标题为“灰度摄像头”
        cv2.imshow("灰度摄像头", gray_frame)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

'''
    8. 图像通道分离与合并(BGR/RGB)
    示例: split_and_merge_channels("D:\\vision\\cv\\test1\\pic\\football.jpg")
'''
def split_and_merge_channels(image_path):
    """
    读取彩色图像，分离 B、G、R 三个通道并分别显示，
    然后将通道重新合并恢复原图并显示。
    """
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"无法读取图像: {image_path}")
        return

    # 分离 B, G, R 通道（OpenCV 默认是 BGR 顺序）
    b, g, r = cv2.split(img)

    # 创建单通道灰度图用于显示（实际仍是 uint8 单通道）
    # 注意：直接显示 b/g/r 是灰度图，亮度反映对应通道强度

    # 显示各个通道
    cv2.imshow("Blue Channel", b)
    cv2.imshow("Green Channel", g)
    cv2.imshow("Red Channel", r)

    # 合并通道恢复原图
    merged_img = cv2.merge([b, g, r])

    # 显示恢复后的图像（应与原图一致）
    cv2.imshow("Merged Image (Restored)", merged_img)

    # 等待按键关闭所有窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()


'''
    9. 保存摄像头视频到文件
    示例: record_camera_video("D:\\vision\\cv\\test1\\pic\\output.avi")
'''
def record_camera_video(output_path="output.avi", fps=20.0, frame_size=(640, 480)):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头，请检查设备连接。")
        return

    # 设置摄像头分辨率（可选，与 frame_size 保持一致）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])

    # 定义编解码器（FourCC）并创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 常用编码器
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    if not out.isOpened():
        print("无法创建视频写入器，请检查编码器或路径。")
        cap.release()
        return

    print(f"正在录制视频... 按 'q' 停止并保存到 {output_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取视频帧。")
            break

        # 写入帧（注意：必须是 BGR 且尺寸与 frame_size 一致）
        out.write(frame)

        # 实时显示预览
        cv2.imshow("Recording... Press 'q' to stop", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("视频已保存！")


# 示例调用
if __name__ == "__main__":
    record_camera_video(r"D:\vision\cv\test1\pic\output.avi")



