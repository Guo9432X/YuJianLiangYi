# my_code/denoise.py
import cv2
import numpy as np

def adaptive_denoise(image):
    """
    自适应去噪处理：评估图像噪声水平并应用中值或高斯滤波。
    """
    # 1. 噪声评估：通过计算拉普拉斯算子的方差来评估图像清晰度/噪声
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    noise_sigma = cv2.meanStdDev(gray)[1][0][0]
    
    # 2. 根据噪声水平选择滤波器
    # 如果标准差较大，说明可能存在较多噪声（或细节非常丰富），应用滤波
    if noise_sigma > 20:
        # 中值滤波：去除椒盐噪声
        denoised = cv2.medianBlur(image, 3)
        # 高斯滤波：平滑高频噪点
        denoised = cv2.GaussianBlur(denoised, (3, 3), 0)
        return denoised
    
    return image # 噪声水平低，保持原样
