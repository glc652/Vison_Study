import cv2
import barcode
import qrcode
from barcode.writer import ImageWriter

data = "32402118"

# ================= 二维码 =================
qr = qrcode.make(data)
qr.save("qrcode.png")

# ================= Code128 =================
code128 = barcode.get_barcode_class("code128")

barcode128_obj = code128(
    data,
    writer=ImageWriter()
)

barcode128_obj.save("barcode128")

# ================= Code39 =================
Code39 = barcode.get_barcode_class("code39")

barcode39_obj = Code39(
    data,
    writer=ImageWriter(),
    add_checksum=False
)

barcode39_obj.save("barcode39")

print("二维码和条形码生成完成")