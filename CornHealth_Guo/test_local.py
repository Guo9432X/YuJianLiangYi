import cv2
import json
from my_code.dip_enhance import enhance_disease_region
from my_code.agent_post_processor import YoloPostProcessorAgent
from my_code.suggestion_builder import generate_suggestion

def run_my_local_tests():
    print("================== 1. 测试DIP模块 ==================")
    img_path = "test_images/sample_corn_leaf.jpg"
    try:
        img = cv2.imread(img_path)
        img_height, img_width, _ = img.shape
        enhanced_img = enhance_disease_region(img)
        cv2.imwrite("output/enhanced_result.jpg", enhanced_img)
        print(f"✅ DIP增强成功，结果保存在 output/enhanced_result.jpg")
    except Exception as e:
        print(f"❌ DIP增强失败: {e}. 请确保'test_images/sample_corn_leaf.jpg'存在且有效。")
        return

    print("\n================== 2. 测试Agent后处理模块 ==================")
    # 【对接周嘉明】: 他的API应该返回这种格式的JSON
    mock_yolo_output = [
        {"box": [100, 200, 150, 250], "class": "大斑病", "confidence": 0.92},
        {"box": [300, 100, 350, 150], "class": "果穗", "confidence": 0.85},
        {"box": [50, 80, 70, 100], "class": "锈病", "confidence": 0.35}, # 这个会被过滤
        {"box": [110, 210, 130, 230], "class": "大斑病", "confidence": 0.55}, # 这个可能被补偿
    ]
    agent = YoloPostProcessorAgent(image_width=img_width, image_height=img_height)
    optimized_boxes, heatmap_path = agent.execute(mock_yolo_output)
    print("✅ Agent后处理成功。")
    print("  - 优化后检测框:")
    print(json.dumps(optimized_boxes, indent=2, ensure_ascii=False))
    print(f"  - 热力图已保存至: {heatmap_path}")

    print("\n================== 3. 测试防治建议生成模块 ==================")
    # 【对接占传润】: 他需要构建这个context对象
    mock_context_data = {
        "detection_summary": {
            "main_disease": "大斑病",
            "disease_list": [
                {"name": "大斑病", "count": 2, "avg_confidence": 0.73},
            ],
            "health_score": 68
        },
        "environment_context": {
            "location": {"province": "黑龙江省", "city": "哈尔滨市", "county": "双城区"},
            "weather": {"temperature": "22°C", "humidity": "78%", "recent_precipitation": "过去24小时内有阵雨"}
        },
        "user_input": {"soil_type": "黑土", "planting_density": "适中"}
    }
    suggestion, source = generate_suggestion(mock_context_data)
    print(f"✅ 防治建议生成成功 (数据来源: {source})。")
    print("  - 生成的建议内容:")
    print(json.dumps(suggestion, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    run_my_local_tests()