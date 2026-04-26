import cv2
import numpy as np
import json
import os
import requests
from pathlib import Path
from dotenv import load_dotenv

# 加载项目根目录 .env
root_dir = Path(__file__).resolve().parent.parent
load_dotenv(root_dir / ".env")

# 工具函数回退
try:
    from .utils import save_image_to_local
except ImportError:
    def save_image_to_local(image, folder, name):
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = os.path.join(folder, f"{name}.png")
        cv2.imwrite(path, image)
        return path


class DeepSeekLLM:
    """DeepSeek 推理客户端"""
    def __init__(self, api_url="https://api.deepseek.com/v1/chat/completions"):
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        self.api_url = api_url
        if not self.api_key.startswith("sk-"):
            raise ValueError("DEEPSEEK_API_KEY 未配置或无效")

    def generate(self, prompt):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "response_format": {"type": "json_object"}
        }
        resp = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class YoloPostProcessorAgent:
    """YOLO 检测结果后处理，支持规则模式与 ReAct 智能体模式"""
    def __init__(self, image_width, image_height, rules_config=None, use_react=False):
        self.width = image_width
        self.height = image_height
        self.rules = rules_config or {
            "lesion_min_area_ratio": 0.0001,
            "lesion_max_area_ratio": 0.1,
            "high_conf_threshold": 0.7,
            "low_conf_threshold": 0.4,
            "lesion_cluster_distance_ratio": 0.1
        }
        self.use_react = use_react
        if self.use_react:
            try:
                self.llm = DeepSeekLLM()
            except Exception:
                self.use_react = False
                self.llm = None
        else:
            self.llm = None

    def execute(self, yolo_outputs):
        """主入口：返回优化后的框列表与热力图路径"""
        if self.use_react and self.llm is not None:
            boxes = self._react_loop(yolo_outputs)
        else:
            boxes = self._calibrate_confidence(yolo_outputs)
            boxes = self._compensate_small_objects(boxes, yolo_outputs)

        heatmap = self._generate_confidence_heatmap(boxes)
        url = save_image_to_local(heatmap, "output", "heatmap")
        return boxes, url

    # ---------- 规则处理方法 ----------
    def _filter_lesion_area(self, boxes):
        """过滤面积不合理的病斑框"""
        total = self.width * self.height
        min_a = total * self.rules["lesion_min_area_ratio"]
        max_a = total * self.rules["lesion_max_area_ratio"]
        return [b for b in boxes
                if "病" not in b["class"] or
                (min_a < (b["box"][2]-b["box"][0])*(b["box"][3]-b["box"][1]) < max_a)]

    def _calibrate_confidence(self, boxes):
        return self._filter_lesion_area(boxes)

    def _compensate_small_objects(self, current, original):
        return self._compensate_with_params(current, original,
                                            self.rules["low_conf_threshold"],
                                            self.rules["high_conf_threshold"],
                                            self.rules["lesion_cluster_distance_ratio"])

    def _compensate_with_params(self, current, original, low, high, cluster_ratio):
        """靠近高置信框的低置信病斑补偿保留"""
        final = list(current)
        high_conf = [b for b in current if "病" in b["class"] and b["confidence"] > high]
        for box_info in original:
            if low <= box_info["confidence"] < high and "病" in box_info["class"]:
                if any(b["box"] == box_info["box"] for b in final):
                    continue
                for hb in high_conf:
                    if self._box_distance(box_info["box"], hb["box"]) < self.width * cluster_ratio:
                        box_info["note"] = "compensated"
                        final.append(box_info)
                        break
        return final

    @staticmethod
    def _box_distance(boxA, boxB):
        cxA, cyA = (boxA[0]+boxA[2])/2, (boxA[1]+boxA[3])/2
        cxB, cyB = (boxB[0]+boxB[2])/2, (boxB[1]+boxB[3])/2
        return np.sqrt((cxA-cxB)**2 + (cyA-cyB)**2)

    def _generate_confidence_heatmap(self, boxes):
        """生成透明热力图（BGRA）"""
        heatmap = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        for b in boxes:
            x1, y1, x2, y2 = map(int, b["box"])
            conf = b["confidence"]
            color = (0, int(255*(1-conf)), 255)
            alpha = int(50 + 150*conf)
            cv2.rectangle(heatmap, (x1, y1), (x2, y2), (*color, alpha), -1)
        return heatmap

    # ---------- ReAct 循环（静默执行） ----------
    def _summarize_state(self, state):
        boxes = state["boxes"]
        classes = {}
        for b in boxes:
            cls = b["class"]
            classes[cls] = classes.get(cls, 0) + 1
        confs = [b["confidence"] for b in boxes] if boxes else [0]
        return json.dumps({
            "image_width": self.width,
            "image_height": self.height,
            "total_boxes": len(boxes),
            "classes": classes,
            "confidence_stats": {
                "mean": float(np.mean(confs)),
                "min": float(np.min(confs)),
                "max": float(np.max(confs))
            },
            "current_rules": state["rules"],
            "last_actions": state.get("history", [])[-3:]
        }, ensure_ascii=False, indent=2)

    def _execute_action(self, state, action, params):
        if action == "FILTER" and params.get("rule_name") == "lesion_area":
            state["boxes"] = self._filter_lesion_area(state["boxes"])
        elif action == "COMPENSATE":
            low = params.get("low_conf_threshold", self.rules["low_conf_threshold"])
            high = params.get("high_conf_threshold", self.rules["high_conf_threshold"])
            ratio = params.get("cluster_distance_ratio", self.rules["lesion_cluster_distance_ratio"])
            if "original_boxes" in state:
                state["boxes"] = self._compensate_with_params(
                    state["boxes"], state["original_boxes"], low, high, ratio)
        elif action == "ADJUST_RULES":
            for k, v in params.items():
                if k in state["rules"]:
                    state["rules"][k] = v
        state["history"].append({"action": action, "params": params})
        return state

    def _build_react_prompt(self, state):
        return f"""
你是一个玉米病害检测的后处理智能体，专门处理玉米叶片病斑检测框。当前状态如下：
{self._summarize_state(state)}

你的任务是逐步优化检测框列表，最终得到一组合理的病斑框。
你可以执行以下动作：
1. FILTER: 根据病斑面积过滤不合理的框，参数包含 "rule_name":"lesion_area"（面积小于图片总面积万分之一或大于10%的会被移除）。
2. COMPENSATE: 补偿那些靠近高置信度框的低置信度病斑，可指定 low_conf_threshold, high_conf_threshold, cluster_distance_ratio。
3. ADJUST_RULES: 修改当前规则参数，例如 lesion_min_area_ratio 或 high_conf_threshold。
4. FINISH: 当认为当前框集合已足够好时，终止优化。

请根据状态，给出下一步的动作（仅输出 JSON 格式，不要包含其他文字）。
输出示例：{{"action": "FILTER", "params": {{"rule_name": "lesion_area"}}}}
"""

    def _react_loop(self, initial_boxes):
        state = {
            "boxes": list(initial_boxes),
            "rules": self.rules.copy(),
            "history": [],
            "original_boxes": list(initial_boxes)
        }
        for _ in range(5):
            try:
                prompt = self._build_react_prompt(state)
                response = self.llm.generate(prompt)
                action_data = json.loads(response)
                action = action_data.get("action")
                if action == "FINISH":
                    break
                params = action_data.get("params", {})
                state = self._execute_action(state, action, params)
            except Exception:
                break
        return state["boxes"]
