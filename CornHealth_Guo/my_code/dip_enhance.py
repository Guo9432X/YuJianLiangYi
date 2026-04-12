import cv2
import numpy as np

def enhance_disease_region(image):
    """
    对玉米叶片图像进行增强处理，突出病害区域。
    使用 CLAHE (限制对比度自适应直方图均衡化) + 轻微锐化。
    
    Args:
        image: BGR 格式的 numpy 数组 (H, W, 3)
    
    Returns:
        enhanced: 增强后的 BGR 图像
    """
    # 转换为 LAB 色彩空间，仅对 L 通道做 CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 创建 CLAHE 对象 (clipLimit=2.0, tileGridSize=(8,8))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    # 合并通道并转回 BGR
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    # 轻微锐化 (核大小为3)
    kernel_sharpen = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel_sharpen)
    
    return enhanced