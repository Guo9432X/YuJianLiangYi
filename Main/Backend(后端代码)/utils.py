# utils.py
import cv2
import os
import time

def save_image_to_local(image_data, output_dir, filename_prefix):
    """
    保存图像到本地（用于测试或临时存储）
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    timestamp = int(time.time())
    filename = f"{filename_prefix}_{timestamp}.jpg"
    filepath = os.path.join(output_dir, filename)
    
    cv2.imwrite(filepath, image_data)
    return filepath
