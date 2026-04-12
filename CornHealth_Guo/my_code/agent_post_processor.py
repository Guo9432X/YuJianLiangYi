import cv2
import numpy as np
from .utils import save_image_to_local  # 使用相对导入

class YoloPostProcessorAgent:
    def __init__(self, image_width, image_height, rules_config=None):
        self.width = image_width
        self.height = image_height
        # 默认规则配置，后续根据真实数据调整
        # 【对接周嘉明】: 需要根据他的模型输出和数据集统计来优化这些规则。
        self.rules = rules_config or {
            "fruit_ear_max_y_ratio": 0.65,      # 果穗中心点不应高于图片65%的位置
            "lesion_min_area_ratio": 0.0001,    # 病斑面积至少占图片总面积的万分之一
            "lesion_max_area_ratio": 0.1,       # 病斑面积不超过图片总面积的10%
            "high_conf_threshold": 0.7,         # 高置信度阈值
            "low_conf_threshold": 0.4,          # 低置信度补偿阈值
            "lesion_cluster_distance_ratio": 0.1 # 病斑聚集距离为图片宽度的10%
        }

    def execute(self, yolo_outputs):
        """
        主执行函数，串联所有后处理步骤。
        返回：(优化后的boxes列表, 热力图的URL)
        """
        # 1. 置信度校准 (剔除不合理框)
        calibrated_boxes = self._calibrate_confidence(yolo_outputs)
        # 2. 小目标补偿 (捞回部分低置信度框)
        compensated_boxes = self._compensate_small_objects(calibrated_boxes, yolo_outputs)
        # 3. 生成置信度热力图
        heatmap_image = self._generate_confidence_heatmap(compensated_boxes)
        # 4. 保存热力图并获取URL
        heatmap_url = save_image_to_local(heatmap_image, "output", "heatmap")
        
        return compensated_boxes, heatmap_url

    def _calibrate_confidence(self, boxes):
        optimized = []
        total_area = self.width * self.height
        for box_info in boxes:
            box = box_info["box"]
            cls = box_info["class"]
            
            # 规则1：果穗位置过滤
            if cls == "果穗":
                center_y = (box[1] + box[3]) / 2
                if center_y < self.height * self.rules["fruit_ear_max_y_ratio"]:
                    continue # 过滤掉位置过高的果穗
            
            # 规则2：病斑面积过滤
            if "病" in cls:
                area = (box[2] - box[0]) * (box[3] - box[1])
                min_area = total_area * self.rules["lesion_min_area_ratio"]
                max_area = total_area * self.rules["lesion_max_area_ratio"]
                if not (min_area < area < max_area):
                    continue # 过滤掉面积过小或过大的病斑
                    
            optimized.append(box_info)
        return optimized

    def _compensate_small_objects(self, current_boxes, original_boxes):
        final_boxes = list(current_boxes)
        high_conf_lesions = [b for b in current_boxes if "病" in b["class"] and b["confidence"] > self.rules["high_conf_threshold"]]
        
        for box_info in original_boxes:
            if self.rules["low_conf_threshold"] <= box_info["confidence"] < self.rules["high_conf_threshold"] and "病" in box_info["class"]:
                # 如果这个框已经被保留，则跳过
                if any(b["box"] == box_info["box"] for b in final_boxes):
                    continue
                
                # 规则：如果一个中低置信度的病斑靠近一个高置信度病斑，则补偿性保留
                for high_conf_box in high_conf_lesions:
                    if self._box_distance(box_info["box"], high_conf_box["box"]) < self.width * self.rules["lesion_cluster_distance_ratio"]:
                        box_info["note"] = "compensated" # 添加标记
                        final_boxes.append(box_info)
                        break
        return final_boxes

    def _box_distance(self, boxA, boxB):
        cxA, cyA = (boxA[0] + boxA[2]) / 2, (boxA[1] + boxA[3]) / 2
        cxB, cyB = (boxB[0] + boxB[2]) / 2, (boxB[1] + boxB[3]) / 2
        return np.sqrt((cxA - cxB)**2 + (cyA - cyB)**2)

    def _generate_confidence_heatmap(self, boxes):
        # 创建一个 BGRA 格式的透明图层
        heatmap = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        for box_info in boxes:
            box = [int(c) for c in box_info["box"]]
            conf = box_info["confidence"]
            
            # 颜色插值：低置信度(黄) -> 高置信度(红)
            # BGR_color = (Blue, Green, Red)
            color = (0, int(255 * (1 - conf)), 255)
            alpha = int(50 + 150 * conf)  # 透明度插值
            
            # 直接在 heatmap 上绘制半透明矩形（多次绘制会自然叠加）
            # 注意：OpenCV 的 rectangle 支持 BGRA 颜色，只需传入 (B,G,R,A)
            cv2.rectangle(heatmap, (box[0], box[1]), (box[2], box[3]), (*color, alpha), -1)
            
        return heatmap