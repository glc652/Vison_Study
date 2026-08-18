import cv2
import numpy as np

def nothing(x):
    pass

# 读取视频第一帧或图片进行调试
video_path = r"D:\vision\mv\HSV\demo_video.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Error: Cannot read video")
    exit()

# 创建窗口
cv2.namedWindow('HSV Calibration')

# 创建滑块调整 HSV 范围
cv2.createTrackbar('H_Lower', 'HSV Calibration', 0, 180, nothing)
cv2.createTrackbar('H_Upper', 'HSV Calibration', 10, 180, nothing)
cv2.createTrackbar('S_Lower', 'HSV Calibration', 50, 255, nothing)
cv2.createTrackbar('S_Upper', 'HSV Calibration', 255, 255, nothing)
cv2.createTrackbar('V_Lower', 'HSV Calibration', 50, 255, nothing)
cv2.createTrackbar('V_Upper', 'HSV Calibration', 255, 255, nothing)

# 转换为 HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

while True:
    # 获取滑块值
    h_lower = cv2.getTrackbarPos('H_Lower', 'HSV Calibration')
    h_upper = cv2.getTrackbarPos('H_Upper', 'HSV Calibration')
    s_lower = cv2.getTrackbarPos('S_Lower', 'HSV Calibration')
    s_upper = cv2.getTrackbarPos('S_Upper', 'HSV Calibration')
    v_lower = cv2.getTrackbarPos('V_Lower', 'HSV Calibration')
    v_upper = cv2.getTrackbarPos('V_Upper', 'HSV Calibration')

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

    # 显示掩码和结果
    display = np.hstack([frame, result, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
    cv2.imshow('HSV Calibration', display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # 按 's' 保存参数
        print(f"\n保存的 HSV 参数:")
        print(f"lower_hsv = np.array([{h_lower}, {s_lower}, {v_lower}])")
        print(f"upper_hsv = np.array([{h_upper}, {s_upper}, {v_upper}])")

cv2.destroyAllWindows()
