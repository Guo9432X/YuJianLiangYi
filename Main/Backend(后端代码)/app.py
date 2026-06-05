import os
import cv2
import numpy as np
import urllib.request
import requests
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import tempfile
import random

# 导入项目模块
from resize_normalize import resize_and_pad
from dip_enhance import full_dip_enhance
from agent_post_processor import YoloPostProcessorAgent
from suggestion_builder import generate_suggestion
from utils import save_image_to_local
from CornDiseaseDetector import CornDiseaseDetector
from denoise import adaptive_denoise

app = Flask(__name__)

# ================= 环境变量 =================
AMAP_KEY = os.environ.get("AMAP_KEY", "")
VISUAL_CROSSING_API_KEY = os.environ.get("VISUAL_CROSSING_API_KEY", "")

# ================= YOLO 模型加载（懒加载） =================
MODEL_PATH = "best.pt"          # 请替换为实际模型文件名
detector = CornDiseaseDetector(MODEL_PATH)

# 可选：简单的去噪 + 增强组合（如果前端不需要精细控制）
def full_dip_process(image, use_dip=True):
    """
    完整的 DIP 处理（去噪 + 光照增强 + CLAHE+锐化）
    use_dip: 是否启用全部增强，默认 True
    """
    if not use_dip:
        return image
    img = image.copy()
    # 去噪
    img = adaptive_denoise(img)
    # 应用完整DIP（去噪+光照增强+CLAHE+锐化）
    img = full_dip_enhance(img, use_light_enhance=True, use_clahe=True)
    return img

# ================= 外部 API 辅助函数 =================
def get_address_by_coords(latitude, longitude):
    """高德逆地理编码，返回省/市/区/adcode"""
    url = "https://restapi.amap.com/v3/geocode/regeo"
    params = {
        "location": f"{longitude},{latitude}",
        "key": AMAP_KEY,
        "output": "JSON",
        "radius": 1000,
        "extensions": "all"
    }
    try:
        resp = requests.get(url, params=params, timeout=5)
        result = resp.json()
        if result.get("status") == "1":
            addr = result["regeocode"]["addressComponent"]
            return {
                "province": addr.get("province", ""),
                "city": addr.get("city", ""),
                "district": addr.get("district", ""),
                "adcode": addr.get("adcode", "")
            }
    except:
        pass
    return {"province": "", "city": "", "district": "", "adcode": ""}

def get_historical_weather(latitude, longitude, days=30):
    """获取过去 days 天的逐日天气数据（Visual Crossing）"""
    if not VISUAL_CROSSING_API_KEY:
        return []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    url = (f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
           f"{latitude},{longitude}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
           f"?unitGroup=metric&key={VISUAL_CROSSING_API_KEY}&include=days&contentType=json")
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        weather_list = []
        for day in data.get("days", []):
            weather_list.append({
                "date": day["datetime"],
                "temp_max": round(day.get("tempmax", 0), 1),
                "temp_min": round(day.get("tempmin", 0), 1),
                "temp_avg": round(day.get("temp", 0), 1),
                "humidity": round(day.get("humidity", 0), 1),
                "precip": round(day.get("precip", 0), 1),
                "sunshine": round(day.get("sunhours", 0), 1)
            })
        return weather_list
    except:
        return []

def format_weather_detailed(weather_list):
    """将30天天气列表格式化为易读的文本，用于 DeepSeek 提示词"""
    if not weather_list:
        return "无详细历史天气数据"
    # 展示最近7天 + 整体统计（避免 token 过长）
    recent = weather_list[-7:]
    lines = ["过去30天天气详情（最近7天示例）："]
    for day in recent:
        lines.append(f"  {day['date']}: 最高{day['temp_max']}℃, 最低{day['temp_min']}℃, "
                     f"平均{day['temp_avg']}℃, 湿度{day['humidity']}%, 降水{day['precip']}mm, 日照{day['sunshine']}h")
    total_precip = sum(d["precip"] for d in weather_list)
    avg_temp = sum(d["temp_avg"] for d in weather_list) / len(weather_list)
    lines.append(f"过去30天总降水量: {total_precip:.1f}mm, 平均温度: {avg_temp:.1f}℃")
    return "\n".join(lines)

# ================= 核心处理接口 =================
# ================= 核心处理接口 =================
@app.route('/api/process', methods=['POST'])
def process_all():
    tmp_img_path = None
    try:
        data = request.json
        image_url = data.get("image_url")
        use_dip = data.get("use_dip", True)
        latitude = data.get("latitude")
        longitude = data.get("longitude")
        
        # ---------- 新增参数 ----------
        smart_mode = data.get("smart_mode", False)          # 启智模式开关
        soil_type = data.get("soil_type", "")               # 土壤类型
        planting_density = data.get("planting_density", "") # 种植密度
        
        # ----- 1. 获取环境信息（若提供了经纬度） -----
        address = {}
        weather_list = []
        weather_detailed_text = "无详细天气数据"
        if latitude is not None and longitude is not None:
            address = get_address_by_coords(latitude, longitude)
            weather_list = get_historical_weather(latitude, longitude, 30)
            weather_detailed_text = format_weather_detailed(weather_list)
        
        # ----- 2. 下载并预处理图片 -----
        resp = urllib.request.urlopen(image_url)
        img_array = np.asarray(bytearray(resp.read()), dtype="uint8")
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"code": 400, "msg": "图片无效"})
        
        # 尺寸归一化（使用填充模式，自适应填充色）
        if(use_dip):
            img, scale, left, top = resize_and_pad(img, target_size=640, mode='pad', adaptive_fill=True)
        
        # 应用DIP流程（光照增强 + CLAHE+锐化）
        img_processed = full_dip_process(img, use_dip)
        
        # ----- 3. YOLO 检测 -----
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_img_path = tmp.name
        cv2.imwrite(tmp_img_path, img_processed)
        raw_yolo_outputs, orig_w, orig_h = detector.predict_and_format(tmp_img_path)
        
        # ----- 4. 后处理 Agent（根据 smart_mode 决定是否启用 ReAct）-----
        agent = YoloPostProcessorAgent(image_width=orig_w, image_height=orig_h, use_react=smart_mode)
        optimized_boxes, heatmap_path = agent.execute(raw_yolo_outputs)
        
        # ----- 5. 统计病害 -----
        disease_counts = {}
        disease_conf_sum = {}
        for b in optimized_boxes:
            name = b['class']
            if name == "健康玉米":
                continue
            conf = b['confidence']
            disease_counts[name] = disease_counts.get(name, 0) + 1
            disease_conf_sum[name] = disease_conf_sum.get(name, 0.0) + conf

        disease_list = []
        if disease_counts:
            for name, count in disease_counts.items():
                avg_conf = disease_conf_sum[name] / count
                disease_list.append({
                    "name": name,
                    "count": count,
                    "avg_confidence": round(avg_conf, 2)
                })
            disease_list.sort(key=lambda x: x['avg_confidence'], reverse=True)
        else:
            disease_list = [{"name": "健康玉米", "count": 0, "avg_confidence": 1.0}]

        # 健康评分计算
        if optimized_boxes:
            health_score = max(0, 70 - len(optimized_boxes) * 10)
        else:
            health_score = random.randint(70, 90)

        # ----- 6. 构建 context_data 并生成防治建议 -----
        env_info = {
            "location": {
                "province": address.get("province", ""),
                "city": address.get("city", ""),
                "district": address.get("district", "")
            },
            "weather_detailed": weather_detailed_text,
            "weather": {
                "temperature": str(weather_list[-1]["temp_avg"]) if weather_list else "未知",
                "humidity": str(weather_list[-1]["humidity"]) if weather_list else "未知",
                "recent_precipitation": "近30天有降水" if any(d["precip"]>0 for d in weather_list) else "近30天无降水"
            }
        }
        
        context_data = {
            "detection_summary": {
                "main_disease": disease_list[0]['name'] if disease_list else "健康",
                "disease_list": disease_list,
                "health_score": health_score
            },
            "environment_context": env_info,
            "user_input": {
                "soil_type": soil_type,
                "planting_density": planting_density
            }
        }
        suggestion, source = generate_suggestion(context_data)
        
        # ----- 7. 返回结果（增加 smart_mode 回显便于调试）-----
        return jsonify({
            "code": 0,
            "data": {
                "health_score": health_score,
                "suggestion": suggestion,
                "heatmap_url": heatmap_path,
                "optimized_boxes": optimized_boxes,
                "source": source,
                "location": address,
                "weather_detailed": weather_detailed_text,
                "weather_summary": {
                    "recent_30days": weather_list,
                    "current_avg_temp": env_info["weather"]["temperature"],
                    "current_humidity": env_info["weather"]["humidity"]
                } if weather_list else {},
                "smart_mode_used": smart_mode   # 可选，告知前端是否启用了智能模式
            }
        })
        
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)})
    finally:
        if tmp_img_path and os.path.exists(tmp_img_path):
            os.remove(tmp_img_path)

# ================= 其他路由（保持不变） =================
@app.route('/api/get_address', methods=['POST'])
def get_address():
    """根据经纬度获取详细地址（逆地理编码）"""
    try:
        data = request.get_json()
        lat = data.get('latitude')
        lng = data.get('longitude')
        if not lat or not lng:
            return jsonify({"code": 400, "msg": "缺少经纬度参数"})
        addr = get_address_by_coords(lat, lng)
        formatted = f"{addr['province']}{addr['city']}{addr['district']}".strip()
        return jsonify({
            "code": 0,
            "address": formatted,
            "province": addr['province'],
            "city": addr['city'],
            "district": addr['district'],
            "township": "",
            "adcode": addr['adcode']
        })
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)})

@app.route('/api/get_history_weather', methods=['POST'])
def get_weather_history():
    try:
        data = request.get_json()
        lat = data.get('latitude')
        lng = data.get('longitude')
        days = data.get('days', 30)
        if not lat or not lng:
            return jsonify({"code": 400, "msg": "缺少经纬度参数"})
        weather_list = get_historical_weather(lat, lng, days)
        return jsonify({
            "code": 0,
            "data": {
                "location": {"lat": lat, "lng": lng},
                "historical_weather": weather_list,
                "data_source": "visualcrossing",
                "query_time": datetime.now().isoformat()
            }
        })
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)})

@app.route('/api/get_suggestion', methods=['POST'])
def get_suggestion_api():
    try:
        data = request.get_json()
        context_data = {
            "detection_summary": data.get("detection_summary", {}),
            "environment_context": data.get("environment_context", {}),
            "user_input": data.get("user_input", {})
        }
        suggestion, source = generate_suggestion(context_data)
        return jsonify({"code": 0, "data": {"suggestion": suggestion, "source": source}})
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)})

@app.route('/ping', methods=['GET'])
def ping():
    return "pong"

if __name__ == '__main__':
    app.run(debug=True, port=5000)