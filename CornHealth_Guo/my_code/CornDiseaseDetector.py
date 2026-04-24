from ultralytics import YOLO

class CornDiseaseDetector:
    def __init__(self, model_path):
        """
        初始化检测器，只在这里加载一次模型
        """
        print(f"正在加载模型: {model_path} ...")
        self.model = YOLO(model_path)
        # 如果需要，这里还可以初始化其他的配置参数
        
    def predict_and_format(self, image_path):
        """
        执行推理，并转换成后处理 Agent 需要的格式
        """
        results = self.model(image_path)
        result = results[0]
        names = result.names 
        orig_height, orig_width = result.orig_shape
        
        yolo_outputs = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            
            yolo_outputs.append({
                "box": [x1, y1, x2, y2],
                "class": names[cls_id],  # 这里输出的就是你 yaml 里的英文标签
                "confidence": conf
            })
            
        return yolo_outputs, orig_width, orig_height
