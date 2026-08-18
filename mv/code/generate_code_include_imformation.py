from PIL import Image, ImageDraw, ImageFont
import qrcode
import os
input_path = r"D:\vision\mv\code\ball.jpg"
output_path = r"D:\vision\mv\code\ball_out.jpg"
img = Image.open(input_path).convert("RGB")
w, h = img.size
draw = ImageDraw.Draw(img)

balls = [
    ("pingpong", "橙色乒乓球"),
    ("volleyball", "排球"),
    ("soccer", "足球"),
    ("golf", "高尔夫球"),
    ("basketball", "篮球"),
    ("soccer", "足球"),
    ("football", "橄榄球"),
    ("tennis", "网球"),
    ("pingpong", "白色乒乓球"),
]
cols = [1/6, 3/6, 5/6]
rows = [1/6, 3/6, 5/6]
box_w = int(w * 0.26)
box_h = int(h * 0.26)
qr_size = int(w * 0.07)

for idx, (name_en, name_cn) in enumerate(balls):
    row = idx // 3
    col = idx % 3

    cx = int(cols[col] * w)
    cy = int(rows[row] * h)

    # 画球的外框
    x1 = max(0, cx - box_w // 2)
    y1 = max(0, cy - box_h // 2)
    x2 = min(w - 1, cx + box_w // 2)
    y2 = min(h - 1, cy + box_h // 2)

    draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=5)

    # 生成二维码，内容为英文球类名称
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=6,
        border=1
    )
    qr.add_data(name_en)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    qr_img = qr_img.resize((qr_size, qr_size))

    qr_x = x2 -80
    qr_y = y1 + (box_h - qr_size) // 2
    if qr_y + qr_size > h:
        qr_y = h - qr_size - 2
    if qr_y < 0:
        qr_y = 2

    pad = 6
    bg_x1 = max(0, qr_x - pad)
    bg_y1 = max(0, qr_y - pad)
    bg_x2 = min(w, qr_x + qr_size + pad)
    bg_y2 = min(h, qr_y + qr_size + pad)

    draw.rectangle((bg_x1, bg_y1, bg_x2, bg_y2), fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    img.paste(qr_img, (qr_x, qr_y))

    # 添加文字标签
    text = name_en
    text_x = x1
    text_y = y2 + 10
    if text_y + 30 > h:
        text_y = y1 - 35
    draw.text((text_x, text_y), text, fill=(255, 0, 0))
img.save(output_path)
print("已生成：", output_path)