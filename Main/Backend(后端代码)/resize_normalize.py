# resize_normalize.py
import cv2
import numpy as np

def resize_and_pad(image, target_size=640, mode='pad', adaptive_fill=True):
    """
    尺寸归一化：将图像缩放到指定尺寸，支持两种模式：
    - 'pad'   : 保持宽高比，长边缩放到 target_size，短边补灰边，最终尺寸为 (target_size, target_size)
    - 'stretch': 直接拉伸到 (target_size, target_size)，不保持比例
    adaptive_fill: 当 mode='pad' 时，根据原图平均亮度自适应选择填充色（深色图用浅灰，浅色图用深灰）
    """
    h, w = image.shape[:2]
    if mode == 'stretch':
        resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        return resized, 1.0, 0, 0  # 缩放因子1，无偏移

    # mode == 'pad': 保持比例，居中填充
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 计算填充色（自适应）
    if adaptive_fill:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        # 暗图用浅灰色(200,200,200)，亮图用深灰色(50,50,50)
        if mean_brightness < 128:
            fill_color = (200, 200, 200)
        else:
            fill_color = (50, 50, 50)
    else:
        fill_color = (114, 114, 114)  # 默认中性灰

    # 创建目标画布并居中放置
    canvas = np.full((target_size, target_size, 3), fill_color, dtype=np.uint8)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = resized

    return canvas, scale, left, top  # 返回填充后的图像，缩放因子，左上角偏移（用于后续坐标映射）