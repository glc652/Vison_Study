import cv2
from pyzbar.pyzbar import decode


def decode_barcode_qrcode(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("图片读取失败")
        return

    results = decode(img)

    if not results:
        print("未识别到一维码或二维码")
        return

    for obj in results:
        data = obj.data.decode("utf-8")
        code_type = obj.type
        x, y, w, h = obj.rect

        print("类型:", code_type)
        print("内容:", data)
        print("-" * 30)

        # 画框
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 显示文字
        cv2.putText(
            img,
            f"{code_type}: {data}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow("result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


decode_barcode_qrcode(r"D:\vision\mv\code\barcode39.png")