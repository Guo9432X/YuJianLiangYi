import cv2
import os
import time

def save_image_to_local(image_data, output_dir, filename_prefix):
    """
    将numpy图像数据保存到本地，并返回模拟的云端URL。
    在真实后端中，这里会执行上传到云存储的操作。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = int(time.time())
    filename = f"{filename_prefix}_{timestamp}.jpg"
    filepath = os.path.join(output_dir, filename)
    
    cv2.imwrite(filepath, image_data)
    
    # 【对接占传润】: 此处需要替换为真实的云存储上传逻辑。
    # 上传成功后，应返回真实的云存储URL，例如：
    # "cloud://your-env-id.xxxxxxxx/output/heatmap_1684789200.jpg"
    # 目前我们先用本地路径作为占位符。
    print(f"图片已保存至本地: {filepath}")
    return filepath