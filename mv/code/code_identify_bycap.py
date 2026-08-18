import cv2
from pyzbar.pyzbar import decode

cap = cv2.VideoCapture(0)
kernel_size = (8, 8)
while True:
    ret, frame = cap.read()

    if not ret:
        print("摄像头读取失败")
        break
    blurred_frame = frame#cv2.blur(frame, kernel_size)#frame

    results = decode(blurred_frame)

    for obj in results:
        data = obj.data.decode("utf-8")
        code_type = obj.type
        x, y, w, h = obj.rect

        print("类型:", code_type, "内容:", data)

        cv2.rectangle(blurred_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            blurred_frame,f"{code_type}: {data}",(x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 255, 0),2
        )

    cv2.imshow("barcode/qrcode scanner", blurred_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()