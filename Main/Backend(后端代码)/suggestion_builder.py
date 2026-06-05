"""
suggestion_builder.py
玉米病害防治建议生成模块（精简九种病害）
"""

import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

def build_prompt(context_data):
    def format_diseases(disease_list):
        if not disease_list:
            return "未检测到明显病害"
        parts = [f"{d['name']} {d['count']}处 (平均置信度{d['avg_confidence']:.2f})" for d in disease_list]
        return "；".join(parts)

    prompt = (
        "你是一位资深的玉米种植与病害防治专家，拥有二十年的田间诊断经验。"
        "你需要结合病害检测结果、当地气候条件、土壤及田间管理信息，给出专业、具体、可操作性强的防治建议。\n\n"
    )
    prompt += "现在，我有一块玉米田遇到了问题，请你根据以下详细信息进行诊断和指导：\n"
    prompt += f"### 1. 核心检测信息:\n- **主要病害**: {context_data['detection_summary']['main_disease']}\n- **病情概览**: {format_diseases(context_data['detection_summary']['disease_list'])}。\n"
    # 添加健康评分计算规则说明
    prompt += f"- **健康评分**: {context_data['detection_summary']['health_score']} 分。评分规则：当检测到病害时，最高分为70分，每发现一处病斑扣10分；未发现病害时，随机在70-90分之间取值。\n\n"

    env = context_data['environment_context']
    loc = env.get('location', {})
    weather_detailed = env.get('weather_detailed', '无详细天气数据')
    u = context_data['user_input']

    # 地理位置字符串处理
    if isinstance(loc, str):
        location_str = loc
    else:
        province = loc.get('province', '')
        city = loc.get('city', '')
        location_str = f"{province}{city}".strip() or "未知位置"

    prompt += f"### 2. 环境与田间信息:\n- **地理位置**: {location_str}\n- **详细天气数据**:\n{weather_detailed}\n- **田间管理**: 土壤类型 {u.get('soil_type', '未知')}，种植密度 {u.get('planting_density', '未知')}。\n\n"
    prompt += "### 3. 你的任务:\n请根据以上所有信息，提供一份结构清晰、操作性强的防治方案，包括：1.当前病情分析, 2.环境风险评估, 3.防治措施(农业/化学), 4.预防建议。\n\n"
    prompt += "### 4. 输出格式:\n请严格按照以下JSON格式返回，不要包含任何额外解释文字或markdown标记：\n"
    prompt += """{"current_analysis": "...", "risk_assessment": "...", "control_measures": {"agricultural_control": "...", "chemical_control": {"recommendations": [{"agent_name": "...", "usage": "...", "precaution": "..."}]}}, "prevention_tips": "..."}"""
    return prompt

def generate_suggestion(context_data):
    if not DEEPSEEK_API_KEY.startswith("sk-"):
        print("警告：未配置有效的 DeepSeek API Key，将直接返回降级模板。")
        return _get_fallback_suggestion(context_data), "fallback"

    prompt = build_prompt(context_data)
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "response_format": {"type": "json_object"}
    }
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=45)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        suggestion = json.loads(content)
        return suggestion, "deepseek"
    except Exception as e:
        print(f"调用失败，启用降级模板: {e}")
        return _get_fallback_suggestion(context_data), "fallback"

def _get_fallback_suggestion(context_data):
    """当 API 不可用时，从本地九种病害元数据库中返回建议"""
    main_disease = context_data.get("detection_summary", {}).get("main_disease", "未知病害")
    if not main_disease or main_disease == "未知病害":
        disease_list = context_data.get("detection_summary", {}).get("disease_list", [])
        if disease_list:
            main_disease = disease_list[0].get("name", "未知病害")

    # 九种病害的降级模板（精简版，可根据实际模型调整）
    templates = {
        "健康玉米": {
            "current_analysis": "当前玉米植株生长健康，未检测到明显病害症状。",
            "risk_assessment": "目前病害发生风险较低，但需关注气候和田间管理。",
            "control_measures": {
                "agricultural_control": "合理密植，平衡施肥，及时排水，清除病残体。",
                "chemical_control": {"recommendations": []}
            },
            "prevention_tips": "选用抗病品种，播种前种子包衣，高发期前喷保护性杀菌剂。"
        },
        "玉米大斑病": {
            "current_analysis": "检测到玉米大斑病，叶片出现长梭形灰褐色大斑，严重时叶片枯死。",
            "risk_assessment": "温暖潮湿条件极易流行，低洼密植田块风险高。",
            "control_measures": {
                "agricultural_control": "轮作，清除病残体，增施磷钾肥，避免偏施氮肥。",
                "chemical_control": {
                    "recommendations": [
                        {"agent_name": "吡唑醚菌酯·戊唑醇", "usage": "30%悬浮剂30-40毫升/亩", "precaution": "发病初期喷施，重点中下部叶片"},
                        {"agent_name": "丙环唑", "usage": "25%乳油30-40毫升/亩", "precaution": "交替用药，避开花期"}
                    ]
                }
            },
            "prevention_tips": "选用抗病品种（如郑单958），种子包衣，加强监测。"
        },
        "玉米小斑病": {
            "current_analysis": "检测到玉米小斑病，叶片出现椭圆形褐色小斑，数量多。",
            "risk_assessment": "高温高湿下流行极快，可造成严重减产。",
            "control_measures": {
                "agricultural_control": "间作套种，及时去病叶，收获后清园。",
                "chemical_control": {
                    "recommendations": [
                        {"agent_name": "肟菌·戊唑醇", "usage": "75%水分散粒剂15-20克/亩", "precaution": "抽雄灌浆期关键防治，间隔7天"},
                        {"agent_name": "丙环·嘧菌酯", "usage": "18.7%悬乳剂50-60毫升/亩", "precaution": "兼防大斑病、锈病"}
                    ]
                }
            },
            "prevention_tips": "选择抗病品种，合理密度，施足基肥。"
        },
        "灰斑病": {
            "current_analysis": "检测到玉米灰斑病，叶片出现灰色至浅褐色椭圆形病斑。",
            "risk_assessment": "多雨年份易发，中后期危害重。",
            "control_measures": {
                "agricultural_control": "选用抗病品种，雨后排水，清除病残体。",
                "chemical_control": {
                    "recommendations": [
                        {"agent_name": "甲基硫菌灵", "usage": "70%可湿性粉剂70-90克/亩", "precaution": "发病初期、大喇叭口期和吐丝期各一次"},
                        {"agent_name": "苯醚甲环唑", "usage": "10%水分散粒剂35-50克/亩", "precaution": "间隔7-10天"}
                    ]
                }
            },
            "prevention_tips": "合理密植，均衡施肥，注意田间通风。"
        },
        "普通锈病": {
            "current_analysis": "检测到玉米普通锈病，叶片出现黄褐色夏孢子堆。",
            "risk_assessment": "温暖多湿条件流行，气流传播快。",
            "control_measures": {
                "agricultural_control": "增施磷钾肥，及时排水，摘除病叶。",
                "chemical_control": {
                    "recommendations": [
                        {"agent_name": "戊唑醇", "usage": "430克/升悬浮剂15-20毫升/亩", "precaution": "病叶率达5%时施药"},
                        {"agent_name": "吡唑醚菌酯·氟环唑", "usage": "17%悬乳剂40-60毫升/亩", "precaution": "可加芸苔素内酯"}
                    ]
                }
            },
            "prevention_tips": "关注气象预报，雨后及时预防。"
        },
        "南方锈病": {
            "current_analysis": "检测到玉米南方锈病，叶片密布黄褐色疱斑。",
            "risk_assessment": "台风传播，高温高湿易暴发，一类病害。",
            "control_measures": {
                "agricultural_control": "排渍降湿，喷施免疫诱抗剂。",
                "chemical_control": {
                    "recommendations": [
                        {"agent_name": "戊唑醇·肟菌酯", "usage": "75%水分散粒剂15-20克/亩", "precaution": "雨后及时喷雾"},
                        {"agent_name": "吡唑醚菌酯·氟环唑", "usage": "17%悬乳剂40-60毫升/亩", "precaution": "无人机作业，亩液量≥1.5升"}
                    ]
                }
            },
            "prevention_tips": "密切关注台风路径，迟播玉米风险更高。"
        },
        "弯孢霉叶斑病": {
            "current_analysis": "检测到弯孢霉叶斑病，病斑中心灰白、边缘暗褐，有黄色晕圈。",
            "risk_assessment": "高温高湿流行快，近年上升明显。",
            "control_measures": {
                "agricultural_control": "轮作，早播，增施有机肥。",
                "chemical_control": {
                    "recommendations": [
                        {"agent_name": "苯醚甲环唑·丙环唑", "usage": "30%悬乳剂20-30毫升/亩", "precaution": "发病率10%时用药，重点叶背"},
                        {"agent_name": "吡唑醚菌酯", "usage": "250克/升乳油30-40毫升/亩", "precaution": "间隔7-10天，连喷2-3次"}
                    ]
                }
            },
            "prevention_tips": "以抗病品种为主，收获后深翻。"
        },
        "圆斑病": {
            "current_analysis": "检测到圆斑病，叶片近圆形灰白斑，外围黄晕，可危害果穗。",
            "risk_assessment": "抽雄吐丝期遇阴雨易流行。",
            "control_measures": {
                "agricultural_control": "抗病品种，摘除病叶果穗，清除残体。",
                "chemical_control": {
                    "recommendations": [
                        {"agent_name": "丙环·嘧菌酯", "usage": "18.7%悬乳剂50-60毫升/亩", "precaution": "吐丝期保护果穗"},
                        {"agent_name": "戊唑醇", "usage": "430克/升悬浮剂15-20毫升/亩", "precaution": "间隔7-10天"}
                    ]
                }
            },
            "prevention_tips": "种子包衣，及时清园。"
        },
        "褐斑病": {
            "current_analysis": "检测到褐斑病，叶片出现黄褐色圆形小斑，散出褐色粉末。",
            "risk_assessment": "高温多雨、氮肥偏施田块发病重。",
            "control_measures": {
                "agricultural_control": "合理密植，均衡施肥，排水降湿。",
                "chemical_control": {
                    "recommendations": [
                        {"agent_name": "三唑酮", "usage": "15%可湿性粉剂60-80克/亩", "precaution": "发病初期5-7天一次"},
                        {"agent_name": "戊唑醇·苯醚甲环唑", "usage": "40%悬浮剂20-30毫升/亩", "precaution": "兼治其他叶部病害"}
                    ]
                }
            },
            "prevention_tips": "提前喷施吡唑醚菌酯预防。"
        },
        "玉米条斑病": {
            "current_analysis": "检测到玉米条斑病，叶片沿叶脉出现褐色长条状病斑。",
            "risk_assessment": "高温高湿下流行，连作低洼地重。",
            "control_measures": {
                "agricultural_control": "轮作，摘除基部病叶，增施磷钾肥。",
                "chemical_control": {
                    "recommendations": [
                        {"agent_name": "吡唑醚菌酯·戊唑醇", "usage": "30%悬浮剂40-50毫升/亩", "precaution": "发病初期，间隔7-10天"},
                        {"agent_name": "肟菌·戊唑醇", "usage": "75%水分散粒剂15-20克/亩", "precaution": "重点喷中下部叶片"}
                    ]
                }
            },
            "prevention_tips": "选用抗条斑病品种，种子包衣。"
        },
        "未知病害": {
            "current_analysis": "未能识别具体病害，可能多种混合感染。",
            "risk_assessment": "建议实地踏查。",
            "control_measures": {
                "agricultural_control": "通风透光，清除病叶，增施磷钾肥。",
                "chemical_control": {"recommendations": []}
            },
            "prevention_tips": "咨询当地农技专家。"
        }
    }
    return templates.get(main_disease, templates["未知病害"])