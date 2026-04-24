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


#周嘉明部分以下-------------------------------------------------------------------------------------
def adaptive_light_enhance(image: np.ndarray, auto_threshold=100, strength=1.0) -> np.ndarray:
    """
    自适应光照增强模块 (针对整体偏暗/背光图像)
    
    Args:
        image: BGR 格式的 numpy 数组 (H, W, 3)，输入应为 640x640
        auto_threshold: 触发增强的平均亮度阈值 (0-255)，默认 100
        strength: 增强强度 (0.0 ~ 1.0)，控制对比度限制
        
    Returns:
        enhanced_image: 增强后的 BGR 图像；若原图亮度达标则直接返回原图
    """
    # 1. 转换 BGR 到 YUV 空间，分离通道
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)
    
    # 2. 评估全局平均亮度 (基于 Y 通道均值)
    mean_brightness = np.mean(y)
    
    # 3. 自适应判断：如果平均亮度 >= 阈值，说明光照正常，跳过增强避免过曝
    if mean_brightness >= auto_threshold:
        # 可以加上打印用于前端提示或后端调试
        # print(f"[跳过] 当前亮度 {mean_brightness:.2f} >= {auto_threshold}")
        return image
        
    # 4. 执行自适应直方图均衡化 (CLAHE)
    # 依据 strength 参数动态调整 clipLimit (强度 0~1 映射到 clipLimit 1.0~3.0)
    clip_limit = 1.0 + (strength * 2.0)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    
    y_enhanced = clahe.apply(y)
    
    # 5. 合并通道并转换回 BGR 空间
    yuv_enhanced = cv2.merge((y_enhanced, u, v))
    enhanced_image = cv2.cvtColor(yuv_enhanced, cv2.COLOR_YUV2BGR)
    
    return enhanced_image

#周嘉明部分以上-------------------------------------------------------------------------------------
