import cv2
import numpy as np

def nothing(x):
    pass

# 读取视频第一帧进行调试
video_path = r"D:\vision\mv\HSV\demo_video.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Error: Cannot read video")
    exit()

# 创建窗口
cv2.namedWindow('Skin Color Calibration')

# 创建滑块调整肤色 HSV 范围
cv2.createTrackbar('H_Lower', 'Skin Color Calibration', 0, 180, nothing)
cv2.createTrackbar('H_Upper', 'Skin Color Calibration', 20, 180, nothing)
cv2.createTrackbar('S_Lower', 'Skin Color Calibration', 20, 255, nothing)
cv2.createTrackbar('S_Upper', 'Skin Color Calibration', 255, 255, nothing)
cv2.createTrackbar('V_Lower', 'Skin Color Calibration', 70, 255, nothing)
cv2.createTrackbar('V_Upper', 'Skin Color Calibration', 255, 255, nothing)

# 转换为 HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

print("调整滑块找到最佳肤色范围")
print("按 's' 保存参数")
print("按 'q' 退出")

while True:
    # 获取滑块值
    h_lower = cv2.getTrackbarPos('H_Lower', 'Skin Color Calibration')
    h_upper = cv2.getTrackbarPos('H_Upper', 'Skin Color Calibration')
    s_lower = cv2.getTrackbarPos('S_Lower', 'Skin Color Calibration')
    s_upper = cv2.getTrackbarPos('S_Upper', 'Skin Color Calibration')
    v_lower = cv2.getTrackbarPos('V_Lower', 'Skin Color Calibration')
    v_upper = cv2.getTrackbarPos('V_Upper', 'Skin Color Calibration')

    lower = np.array([h_lower, s_lower, v_lower])
    upper = np.array([h_upper, s_upper, v_upper])

    # 创建掩码
    mask = cv2.inRange(hsv, lower, upper)

    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 显示结果
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # 添加文字显示当前参数
    text = f"H: {h_lower}-{h_upper} | S: {s_lower}-{s_upper} | V: {v_lower}-{v_upper}"
    cv2.putText(result, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 显示原图、结果和掩码
    display = np.hstack([frame, result, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
    cv2.imshow('Skin Color Calibration', display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # 按 's' 保存参数
        print(f"\n保存的肤色 HSV 参数:")
        print(f"lower_skin = np.array([{h_lower}, {s_lower}, {v_lower}])")
        print(f"upper_skin = np.array([{h_upper}, {s_upper}, {v_upper}])")
        print("\n将这些值复制到 test4.py 中")

cv2.destroyAllWindows()
