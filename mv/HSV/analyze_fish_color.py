import cv2
import numpy as np
import os
from pathlib import Path

fish_dir = r"D:\vision\mv\HSV\fish"
images = sorted(Path(fish_dir).glob("*.jpg"))

print(f"Found {len(images)} fish images\n")

all_hsv_values = []

for img_path in images:
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Failed to read: {img_path.name}")
        continue

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    h_mean, h_std = h.mean(), h.std()
    s_mean, s_std = s.mean(), s.std()
    v_mean, v_std = v.mean(), v.std()

    h_min, h_max = h.min(), h.max()
    s_min, s_max = s.min(), s.max()
    v_min, v_max = v.min(), v.max()

    print(f"{img_path.name}:")
    print(f"  H: mean={h_mean:.1f}±{h_std:.1f}, range=[{h_min}, {h_max}]")
    print(f"  S: mean={s_mean:.1f}±{s_std:.1f}, range=[{s_min}, {s_max}]")
    print(f"  V: mean={v_mean:.1f}±{v_std:.1f}, range=[{v_min}, {v_max}]")
    print()

    all_hsv_values.append({
        'h_mean': h_mean, 'h_std': h_std, 'h_min': h_min, 'h_max': h_max,
        's_mean': s_mean, 's_std': s_std, 's_min': s_min, 's_max': s_max,
        'v_mean': v_mean, 'v_std': v_std, 'v_min': v_min, 'v_max': v_max
    })

if all_hsv_values:
    print("\n=== 总体统计 ===")
    h_means = [v['h_mean'] for v in all_hsv_values]
    s_means = [v['s_mean'] for v in all_hsv_values]
    v_means = [v['v_mean'] for v in all_hsv_values]

    h_all_min = min([v['h_min'] for v in all_hsv_values])
    h_all_max = max([v['h_max'] for v in all_hsv_values])
    s_all_min = min([v['s_min'] for v in all_hsv_values])
    s_all_max = max([v['s_max'] for v in all_hsv_values])
    v_all_min = min([v['v_min'] for v in all_hsv_values])
    v_all_max = max([v['v_max'] for v in all_hsv_values])

    print(f"H: 平均值={np.mean(h_means):.1f}, 范围=[{h_all_min}, {h_all_max}]")
    print(f"S: 平均值={np.mean(s_means):.1f}, 范围=[{s_all_min}, {s_all_max}]")
    print(f"V: 平均值={np.mean(v_means):.1f}, 范围=[{v_all_min}, {v_all_max}]")

    print(f"\n推荐的 HSV 范围:")
    print(f"lower_hsv = np.array([{max(0, h_all_min-5)}, {max(0, s_all_min-10)}, {max(0, v_all_min-10)}])")
    print(f"upper_hsv = np.array([{min(180, h_all_max+5)}, {min(255, s_all_max+10)}, {min(255, v_all_max+10)}])")

