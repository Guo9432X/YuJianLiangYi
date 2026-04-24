"""
suggestion_builder.py
玉米病害防治建议生成模块

功能：
- generate_suggestion(): 调用 DeepSeek API 实时生成防治建议（主流程）
- _get_fallback_suggestion(): 当 API 不可用时，基于内置的病害元数据库返回专业静态建议（降级方案）
- build_prompt(): 构建发送给 DeepSeek 的专业提示词

接口说明：
- 所有函数签名与原有版本完全兼容，确保项目其他模块不受影响。
- 返回的字典结构固定，便于前端解析。
"""

import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

# ================= 配置（与原有代码一致） =================
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"


# ================= 1. 构建 AI 提示词（原有逻辑，保持不变） =================
def build_prompt(context_data):
    def format_diseases(disease_list):
        if not disease_list:
            return "未检测到明显病害"
        parts = [f"{d['name']} {d['count']}处 (平均置信度{d['avg_confidence']:.2f})" for d in disease_list]
        return "；".join(parts)

    prompt = (
        "你是一位资深的玉米种植与病害防治专家，拥有二十年的田间诊断经验，精通玉米大斑病、锈病、灰斑病、茎腐病等常见病害的识别与综合防治。"
        "你需要结合病害检测结果、当地气候条件、土壤及田间管理信息，给出专业、具体、可操作性强的防治建议。\n\n"
    )
    prompt += "现在，我有一块玉米田遇到了问题，请你根据以下详细信息进行诊断和指导：\n"
    prompt += f"### 1. 核心检测信息:\n- **主要病害**: {context_data['detection_summary']['main_disease']}\n- **病情概览**: {format_diseases(context_data['detection_summary']['disease_list'])}。\n- **健康评分**: {context_data['detection_summary']['health_score']} 分。\n\n"

    loc = context_data['environment_context']['location']
    w = context_data['environment_context']['weather']
    u = context_data['user_input']

    prompt += f"### 2. 环境与田间信息:\n- **地理位置**: {loc.get('province', '')} {loc.get('city', '')}\n- **气候条件**: 温度 {w.get('temperature', '未知')}，湿度 {w.get('humidity', '未知')}。{w.get('recent_precipitation', '近期降水未知')}。\n- **田间管理**: 土壤类型 {u.get('soil_type', '未知')}，种植密度 {u.get('planting_density', '未知')}。\n\n"
    prompt += "### 3. 你的任务:\n请根据以上所有信息，提供一份结构清晰、操作性强的防治方案，包括：1.当前病情分析, 2.环境风险评估, 3.防治措施(农业/化学), 4.预防建议。\n\n"
    prompt += "### 4. 输出格式:\n请严格按照以下JSON格式返回，不要包含任何额外解释文字或markdown标记：\n"
    prompt += """{"current_analysis": "...", "risk_assessment": "...", "control_measures": {"agricultural_control": "...", "chemical_control": {"recommendations": [{"agent_name": "...", "usage": "...", "precaution": "..."}]}}, "prevention_tips": "..."}"""
    return prompt


# ================= 2. 主生成函数（原有逻辑，保持不变） =================
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
    except requests.exceptions.HTTPError as e:
        print(f"HTTP 错误: {e.response.status_code} - {e.response.text}")
        return _get_fallback_suggestion(context_data), "fallback"
    except Exception as e:
        print(f"DeepSeek API调用失败，启用降级模板。错误: {e}")
        return _get_fallback_suggestion(context_data), "fallback"


# ================= 3. 降级方案（全面覆盖玉米常见病害，接口完全不变） =================
def _get_fallback_suggestion(context_data):
    """
    当 DeepSeek API 不可用时，从本地病害元数据库中返回专业建议。
    参数 context_data 与 generate_suggestion 完全相同，从中提取主要病害名称。
    返回的字典结构与 AI 生成的结构完全一致。
    """
    # 提取主要病害名称（兼容多种可能的字段名）
    main_disease = context_data.get("detection_summary", {}).get("main_disease", "未知病害")
    if not main_disease or main_disease == "未知病害":
        disease_list = context_data.get("detection_summary", {}).get("disease_list", [])
        if disease_list:
            main_disease = disease_list[0].get("name", "未知病害")

    # ========== 病害元数据库（覆盖图片所列 15 种病害 + 健康玉米） ==========
    templates = {
        # 0. 健康玉米
        "健康玉米": {
            "current_analysis": "当前玉米植株生长健康，未检测到明显病害症状。叶片颜色正常，无病斑、霉层或坏死组织，整体长势良好。",
            "risk_assessment": "目前病害发生风险较低，但仍需关注气候条件和田间管理。若未来出现连续阴雨、高湿或高温天气，某些病害可能零星发生。",
            "control_measures": {
                "agricultural_control": "保持合理种植密度，及时中耕除草；平衡施肥，增施有机肥和磷钾肥；注意排灌，避免田间积水；定期巡田，发现病株立即清除。",
                "chemical_control": {"recommendations": []}
            },
            "prevention_tips": "选用抗病品种，播种前进行种子包衣处理；收获后彻底清除田间病残体并深翻；根据当地植保部门的预报，在病害高发期前喷施保护性杀菌剂（如代森锰锌）进行预防。"
        },
        # 1. 灰斑病
        "灰斑病": {
            "current_analysis": "检测到玉米灰斑病。主要危害叶片、叶鞘和苞叶，在玉米生长中后期最为活跃。初在叶面上形成无明显边缘的椭圆形至矩圆形灰色至浅褐色病斑，多数沿叶脉扩展，后期病斑上散生许多黑色小点，严重时叶片干枯。",
            "risk_assessment": "灰斑病是近年上升很快、危害较严重的叶部病害之一。病菌通过病残体越冬，分生孢子可重复侵染，7-8月多雨天气易发病。",
            "control_measures": {
                "agricultural_control": "选用抗病品种；合理密植，科学施肥；雨后及时排水，防止田间湿气滞留；收获后清除病残体。",
                "chemical_control": {
                    "recommendations": [
                        {"agent_name": "甲基硫菌灵", "usage": "70%可湿性粉剂70-90克/亩", "precaution": "分别于发病初期、大喇叭口期和抽雄吐丝期进行药剂防治"},
                        {"agent_name": "苯醚甲环唑", "usage": "10%水分散粒剂35-50克/亩", "precaution": "每隔7-10天喷施1次，共防治2-3次"}
                    ]
                }
            },
            "prevention_tips": "灰斑病防治可采取“三步走”策略：先清除严重病叶减少菌源，再喷施组合药剂，最后加强排水和追肥管理。"
        },
        # 2. 玉米条斑病
        "玉米条斑病": {
            "current_analysis": "检测到玉米条斑病。该病主要危害叶片，病斑沿叶脉方向延伸呈长条状，初期为水渍状暗绿色小点，后变为褐色或紫褐色，多个条斑可融合成片，导致叶片提早枯死。",
            "risk_assessment": "高温高湿（25-32℃）条件下易流行。病菌随病残体在土壤中越冬，分生孢子借风雨传播，连作田、低洼积水田块发病重。",
            "control_measures": {
                "agricultural_control": "实行轮作倒茬；合理密植，改善通风透光；增施磷钾肥，避免偏施氮肥；及时摘除基部病叶并带出田间。",
                "chemical_control": {
                    "recommendations": [
                        {"agent_name": "吡唑醚菌酯·戊唑醇", "usage": "30%悬浮剂40-50毫升/亩兑水30公斤喷雾", "precaution": "发病初期施药，间隔7-10天，连续2-3次"},
                        {"agent_name": "肟菌·戊唑醇", "usage": "75%水分散粒剂15-20克/亩", "precaution": "重点喷施中下部叶片，注意轮换用药"}
                    ]
                }
            },
            "prevention_tips": "选用抗条斑病品种；播种前进行种子包衣；收获后彻底清除病残体，减少初侵染源。"
        },
        # 3. 褐斑病
        "褐斑病": {
            "current_analysis": "检测到玉米褐斑病。该病由玉蜀黍节壶菌引起，主要危害果穗以下叶片和叶鞘。最初为白色至黄色小斑，渐渐变成黄褐色或红褐色，圆形或椭圆形，后期叶片上可布满病斑，散出褐色粉末，造成叶片局部或全叶干枯。一般减产10%左右，严重时达30%以上。",
            "risk_assessment": "褐斑病在高温多雨季节易暴发流行。田间湿度大、通风透光差、氮肥偏施的田块发病风险更高。该病常与锈病、大斑病等叶部病害混合发生。",
            "control_measures": {
                "agricultural_control": "合理密植，改善田间通风透光条件；均衡施肥，避免偏施氮肥；雨后及时排水，降低田间湿度；清除田间病残体。",
                "chemical_control": {
                    "recommendations": [
                        {"agent_name": "三唑酮（粉锈宁）", "usage": "15%可湿性粉剂60-80克/亩兑水喷雾", "precaution": "发病初期及时用药，间隔5-7天连续喷施2-3次"},
                        {"agent_name": "戊唑醇·苯醚甲环唑", "usage": "40%悬浮剂20-30毫升/亩", "precaution": "可一同喷施叶面肥促进玉米生长，兼治其他叶部病害"}
                    ]
                }
            },
            "prevention_tips": "没有发病的田块可提前喷施吡唑醚菌酯进行预防，同时可预防锈病。喷药后6小时内如遇雨应及时补喷。"
        },
        # 4. 普通锈病
        "普通锈病": {
            "current_analysis": "检测到玉米普通锈病。此病害主要危害叶片，严重时也侵染叶鞘和苞叶。叶片上出现散生或聚生的黄褐色夏孢子堆（突起疱斑），破裂后散出锈黄色粉末，导致叶片干枯死亡。轻者减产10-20%，重者达30%以上。",
            "risk_assessment": "温暖、多湿环境有利于锈病流行。高温高湿、降雨频繁时病害可暴发成灾。该病通过气流传播，具有传播流行速度快、暴发性强的特点。",
            "control_measures": {
                "agricultural_control": "选用抗病品种；加强水肥管理，增施磷钾肥；雨后及时排水，降低田间湿度；零星发病时及时摘除病叶。",
                "chemical_control": {
                    "recommendations": [
                        {"agent_name": "戊唑醇", "usage": "430克/升悬浮剂15-20毫升/亩兑水30公斤喷雾", "precaution": "防治适期为大喇叭口期到吐丝期，病叶率达5%时立即施药"},
                        {"agent_name": "吡唑醚菌酯·氟环唑", "usage": "17%悬乳剂40-60毫升/亩", "precaution": "视发病情况隔7-10天再施药1次，可添加芸苔素内酯增强抗逆性"}
                    ]
                }
            },
            "prevention_tips": "密切关注天气变化和病害测报信息，在降雨过后及时施药预防；玉米收获后彻底清除田间病残体。"
        },
        # 5. 南方锈病
        "南方锈病": {
            "current_analysis": "检测到玉米南方锈病。这是农业农村部公布的一类农作物病虫害，具有传播流行速度快、暴发性强、危害损失重的特点。叶片上初生褪绿小斑点，很快发展为黄褐色突起疱斑，分布密集。严重时叶片布满孢子堆，导致提前枯死。",
            "risk_assessment": "南方锈病通过台风等气流进行远距离传播，高温（24-28℃）、多雨、高湿条件适于病害发生。年度间发生差异与台风活动路径密切相关。",
            "control_measures": {
                "agricultural_control": "合理施用氮肥，增施磷、钾肥；积水严重田块及时排水除渍；长势偏弱的玉米可适时喷施免疫诱抗或生长调节剂。",
                "chemical_control": {
                    "recommendations": [
                        {"agent_name": "戊唑醇·肟菌酯", "usage": "75%水分散粒剂15-20克/亩", "precaution": "在降雨过后及时叶面喷雾防治，视病情隔7-10天再次施药"},
                        {"agent_name": "吡唑醚菌酯·氟环唑", "usage": "17%悬乳剂40-60毫升/亩", "precaution": "建议使用植保无人机作业，亩喷液量不低于1.5升"}
                    ]
                }
            },
            "prevention_tips": "密切关注台风路径和病害预警信息；较幼嫩叶片更易感病，迟播玉米在有发病条件时病害可能重于早播玉米。"
        },
        # 6. 大斑病
        "大斑病": {
            "current_analysis": "检测到玉米大斑病。该病主要危害叶片，病斑初期为水渍状青灰色小点，后沿叶脉迅速扩展为长梭形或边缘不整齐的大斑，中央灰褐色，边缘深褐色，潮湿时病斑表面密生黑色霉层。严重时多个病斑连片，叶片枯死。",
            "risk_assessment": "温暖潮湿（20-25℃、相对湿度90%以上）条件下易流行。病菌以菌丝或分生孢子附着在病残体上越冬，分生孢子借风雨传播。连作地、低洼地、种植密度过大、偏施氮肥的田块发病重。",
            "control_measures": {
                "agricultural_control": "实行2-3年轮作；合理密植，及时中耕排湿；增施腐熟有机肥和磷钾肥，避免偏施氮肥；收获后彻底清除病残体，深翻土壤。",
                "chemical_control": {
                    "recommendations": [
                        {
                            "agent_name": "吡唑醚菌酯·戊唑醇",
                            "usage": "30%悬浮剂30-40毫升/亩",
                            "precaution": "发病初期（下部叶片出现病斑时）施药，重点喷施中下部叶片，间隔7-10天，连喷2次"
                        },
                        {
                            "agent_name": "丙环唑",
                            "usage": "25%乳油30-40毫升/亩",
                            "precaution": "可与代森锰锌、百菌清交替使用，延缓抗药性，注意避开花期"
                        }
                    ]
                }
            },
            "prevention_tips": "选用抗病品种（如郑单958、先玉335等）；播种前用戊唑醇或咯菌腈进行种子包衣；加强田间监测，发现中心病株及时拔除并喷药保护。"
        }
        # 7. 小斑病
        "小斑病": {
            "current_analysis": "检测到玉米小斑病。整个生育期均可发生，主要危害叶片、叶鞘和苞叶，严重时可导致果穗腐烂。叶片上病斑较小但数量多，呈椭圆形褐色，边缘赤褐色，潮湿时产生暗黑色霉层。",
            "risk_assessment": "高温高湿（28-32℃）条件下极易流行。病菌侵入后仅需2-4天即可完成一次侵染循环，传播速度快、危害重，一般减产15-20%，严重时可达50%以上。",
            "control_measures": {
                "agricultural_control": "选用抗病品种；实行间作套种，合理密植；施足基肥，适时追肥；生长期及时去除病叶，收获后清除田间病残体。",
                "chemical_control": {
                    "recommendations": [
                        {"agent_name": "肟菌·戊唑醇", "usage": "75%水分散粒剂15-20克/亩兑水30公斤喷雾", "precaution": "抽雄灌浆期为防治关键期，每隔7天喷1次，连续2-3次"},
                        {"agent_name": "丙环·嘧菌酯", "usage": "18.7%悬乳剂50-60毫升/亩喷雾", "precaution": "可兼防大斑病、锈病，施药后6小时内遇雨需补喷"}
                    ]
                }
            },
            "prevention_tips": "本地品种一般比引进品种抗病，白粒型比黄粒型抗病，品种选择时应予以考虑。"
        },
        # 8. 弯孢霉叶斑病
        "弯孢霉叶斑病": {
            "current_analysis": "检测到玉米弯孢霉叶斑病（又称黑霉病）。主要发生在玉米生长中后期，病斑初为水渍状褪绿小点，扩展后呈圆形或椭圆形，中心黄白色至灰白色，边缘暗褐色，外围有黄色晕圈（“三圈状”特征），湿度大时产生灰黑色霉层。一般减产20-30%，严重时减产50%以上。",
            "risk_assessment": "高温高湿（28-32℃）环境利于病害发生。病菌通过病残体越冬，分生孢子靠风雨传播，可引起多次再侵染。该病是近年上升较快、必须高度重视的叶部病害。",
            "control_measures": {
                "agricultural_control": "选用抗病品种（如农大108、郑单14等）；轮作倒茬，适期早播；施用腐熟有机肥，增施磷钾肥。",
                "chemical_control": {
                    "recommendations": [
                        {"agent_name": "苯醚甲环唑·丙环唑", "usage": "30%悬乳剂20-30毫升/亩", "precaution": "发病初期（发病率达10%时）及时用药，重点喷洒叶片背面"},
                        {"agent_name": "吡唑醚菌酯", "usage": "250克/升乳油30-40毫升/亩", "precaution": "每隔7-10天喷1次，连喷2-3次，注意轮换用药"}
                    ]
                }
            },
            "prevention_tips": "防治策略以种植抗病品种为主、药剂防治为辅。收获后及时清除田间病残体并进行深翻，减少来年菌源。"
        },
        # 9. 圆斑病
        "圆斑病": {
            "current_analysis": "检测到玉米圆斑病。主要危害叶片、叶鞘和苞叶，也可侵染果穗。叶片上病斑近圆形，直径1-2厘米，中央灰白色，边缘褐色，外围有黄色晕圈，病斑上可产生黑色霉层。果穗受害后籽粒变黑、腐烂。",
            "risk_assessment": "该病由玉米圆斑病菌（蠕孢菌）引起，分生孢子借风雨传播。抽雄吐丝期若遇连续阴雨、高湿条件，病害易流行。感病品种、密植田块发病重。",
            "control_measures": {
                "agricultural_control": "选用抗病品种；合理轮作；加强水肥管理，增施磷钾肥；及时摘除病叶、病果穗并带出田间销毁。",
                "chemical_control": {
                    "recommendations": [
                        {"agent_name": "丙环·嘧菌酯", "usage": "18.7%悬乳剂50-60毫升/亩", "precaution": "抽雄吐丝期施药保护果穗，重点喷施穗部和上部叶片"},
                        {"agent_name": "戊唑醇", "usage": "430克/升悬浮剂15-20毫升/亩", "precaution": "视病情隔7-10天再施1次，注意轮换用药"}
                    ]
                }
            },
            "prevention_tips": "选用抗圆斑病品种；播种前进行种子包衣处理；收获后及时清除病残体，减少越冬菌源。"
        },
        # 未知病害
        "未知病害": {
            "current_analysis": "未能识别出具体病害类型，或检测到多种混合感染。当前症状可能不典型，或由多种病原复合侵染所致。",
            "risk_assessment": "请密切关注植株状态，防止病情蔓延。建议实地踏查，观察病害发展动态。",
            "control_measures": {
                "agricultural_control": "保持田间通风透光，及时清理病叶、病株；合理灌溉，避免田间积水；增施磷钾肥提高植株抗病力。",
                "chemical_control": {"recommendations": []}
            },
            "prevention_tips": "建议咨询当地农技服务中心或植保专家进行实地诊断，以获得更精确的病害识别和防治方案。同时注意观察病害发生部位、病斑形态、霉层颜色等特征，便于后续精准识别。"
        }
    }

    # 返回匹配的模板，若找不到则返回默认的“未知病害”
    return templates.get(main_disease, templates["未知病害"])
