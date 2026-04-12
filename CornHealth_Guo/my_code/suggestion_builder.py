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
        # 如果未指定，尝试从 disease_list 中取第一个
        disease_list = context_data.get("detection_summary", {}).get("disease_list", [])
        if disease_list:
            main_disease = disease_list[0].get("name", "未知病害")

    # ========== 病害元数据库（覆盖 14+ 种玉米主要病害） ==========
    templates = {
        # ---------- 叶斑类 ----------
        "大斑病": {
            "current_analysis": "检测到玉米大斑病。该病主要危害叶片，从下部老叶开始发病，病斑呈长梭形、灰褐色或黄褐色，严重时多个病斑连接导致叶片大面积枯死，严重影响光合作用和产量。",
            "risk_assessment": "中温（26-30℃）、高湿环境下易流行。若近期持续阴雨或多雾，田间湿度大，病害存在快速扩散风险。连作地、密植田块发病风险更高。",
            "control_measures": {
                "agricultural_control": "合理轮作，避免连作；合理密植，改善田间通风透光条件；增施磷钾肥，避免偏施氮肥；及时摘除病叶并带出田间销毁，收获后深翻灭茬。",
                "chemical_control": {
                    "recommendations": [
                        {"agent_name": "苯醚甲环唑", "usage": "发病初期使用10%水分散粒剂1500倍液均匀喷雾", "precaution": "间隔7-10天，连续2-3次，注意轮换用药"},
                        {"agent_name": "吡唑醚菌酯·戊唑醇", "usage": "30%悬浮剂40-50毫升/亩兑水30公斤喷雾", "precaution": "重点喷施中下部叶片，避免高温时段施药"}
                    ]
                }
            },
            "prevention_tips": "选用抗病品种（如农大108、郑单958等）是最经济有效的预防措施；播种前彻底清除田间病残体，减少初侵染源。"
        },
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
        "锈病": {
            "current_analysis": "检测到玉米锈病。此病害主要危害叶片，严重时也侵染叶鞘和苞叶。叶片上出现散生或聚生的黄色至黄褐色夏孢子堆（突起疱斑），破裂后散出锈黄色粉末，导致叶片干枯死亡。轻者减产10-20%，重者达30%以上。",
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
        "白斑病": {
            "current_analysis": "检测到玉米白斑病。这是近年来新发突发的玉米叶部病害。通常在抽雄吐丝期发病，初期在叶片上出现豌豆大小水渍状褪绿病斑，随后病斑逐渐变为白色或灰白色，边缘清晰，后期病斑可连接成片，导致叶片枯萎、整株枯死。一般产量损失10%-50%，严重时可致绝收。",
            "risk_assessment": "白斑病发生区域呈现快速扩张趋势，对玉米安全生产构成严重威胁。病害传播迅速，常被误认为除草剂药害斑而延误防治。",
            "control_measures": {
                "agricultural_control": "选用抗病品种；合理密植，加强田间通风；科学施肥，增施磷钾肥；及时清除田间病残体。",
                "chemical_control": {
                    "recommendations": [
                        {"agent_name": "唑醚·氟环唑", "usage": "17%悬乳剂40-60毫升/亩", "precaution": "抓住病害发生初期及时喷施，根据病情严重程度决定施药次数"},
                        {"agent_name": "丁香·戊唑醇", "usage": "20%悬浮剂30-50毫升/亩", "precaution": "注意轮换用药，避免抗药性产生"}
                    ]
                }
            },
            "prevention_tips": "优先采用农业防治措施，科学合理选用高效低毒农药，避免盲目用药。密切关注当地植保部门发布的病害预警信息。"
        },
        "北方炭疽病": {
            "current_analysis": "检测到玉米北方炭疽病。由玉蜀黍球梗孢菌引起，在玉米各生育期均可发生。主要危害叶片，严重时也危害叶鞘、苞叶，引起根腐、茎腐、顶腐。被侵染叶片布满小型圆形或椭圆形病斑，病斑密集时可导致大片的叶片组织坏死。发生严重时可导致产量损失超过50%。",
            "risk_assessment": "低温高湿条件利于病害发生，北方春玉米区为重点防控区域。病菌可造成玉米叶肉组织大面积坏死，同时降低玉米平均株高，对产量影响较大。",
            "control_measures": {
                "agricultural_control": "选用抗（耐）病品种；合理密植，科学施肥、排灌；收获后及时清除田间病残体，深翻土壤。",
                "chemical_control": {
                    "recommendations": [
                        {"agent_name": "吡唑醚菌酯", "usage": "250克/升乳油30-40毫升/亩（预防）", "precaution": "甲氧基丙烯酸酯类药剂预防效果更好，三唑类治疗效果更好"},
                        {"agent_name": "氟环唑·吡唑醚菌酯", "usage": "17%悬乳剂40-60毫升/亩", "precaution": "可用生物制剂与三唑类药剂混用，兼防大斑病、灰斑病等"}
                    ]
                }
            },
            "prevention_tips": "北方炭疽病在发病初期与大斑病症状较相似，需重点观察病斑形状、颜色及霉层特征，准确识别病害是科学防治的前提。"
        },
        # ---------- 茎秆与根部病害 ----------
        "茎腐病": {
            "current_analysis": "检测到玉米茎腐病（又称青枯病/茎基腐病）。这是典型的土传真菌病害，多在玉米生长中后期发生。典型症状是植株茎基部腐烂、青枯、倒伏，果穗尚未成熟便开始下垂。病株叶片自下而上迅速枯死，茎基部初为水浸状，后逐渐变为淡褐色，手捏有空心感。发病迅速，严重时减产可达80%以上。",
            "risk_assessment": "茎腐病与土壤带菌、地下害虫危害及田间积水密切相关。乳熟期至蜡熟期为发病高峰期，连续阴雨、田间积水的田块发病风险极高。",
            "control_measures": {
                "agricultural_control": "与非禾本科作物轮作2年以上；增施农家肥和钾肥（如硫酸钾）；采用高垄栽培，雨后及时排水；及时拔除病株。",
                "chemical_control": {
                    "recommendations": [
                        {"agent_name": "精甲·咯菌腈", "usage": "6.25%悬浮种衣剂按药种比1:200-300包衣", "precaution": "播种前种子处理是最经济有效的预防措施"},
                        {"agent_name": "恶霉灵·霜霉威", "usage": "30%水剂800-1000倍液灌根", "precaution": "发病初期进行茎基部喷淋或灌根处理"}
                    ]
                }
            },
            "prevention_tips": "选用抗病品种，增施钾肥可显著增强植株抗病力；及时防治地下害虫减少伤口侵染。"
        },
        "纹枯病": {
            "current_analysis": "检测到玉米纹枯病。主要危害叶鞘、叶片和果穗。叶鞘上出现水渍状暗绿色病斑，后扩大成云纹状不规则斑，边缘褐色，中央淡褐色，病部可产生白色菌丝和褐色菌核。严重时叶片枯死，果穗霉变。",
            "risk_assessment": "高温高湿、种植密度大、通风透光差的田块易发病。菌核在土壤中可存活多年，连作田发病逐年加重。",
            "control_measures": {
                "agricultural_control": "合理轮作；清除田间菌源；合理密植，改善通风透光；增施磷钾肥，避免偏施氮肥。",
                "chemical_control": {
                    "recommendations": [
                        {"agent_name": "井冈霉素", "usage": "5%水剂200-250毫升/亩", "precaution": "发病初期喷施茎基部，重点喷施叶鞘部位"},
                        {"agent_name": "苯醚甲环唑·丙环唑", "usage": "30%悬乳剂20-30毫升/亩", "precaution": "隔7-10天再喷1次，连续2-3次"}
                    ]
                }
            },
            "prevention_tips": "选用抗病品种；清除田间病残体和菌核；避免田间积水。"
        },
        # ---------- 穗部与粒部病害 ----------
        "丝黑穗病": {
            "current_analysis": "检测到玉米丝黑穗病。该病为系统性侵染病害，苗期症状不明显，抽穗后才显现。病株果穗变成一团黑粉包（孢子堆），外被白色薄膜，破裂后散出大量黑粉，仅残留维管束丝状物。一般发病率等于损失率。",
            "risk_assessment": "病菌以厚垣孢子在土壤、粪肥或种子表面越冬，土壤带菌是主要侵染来源。连作田、播种过深、地温低的田块发病重。",
            "control_measures": {
                "agricultural_control": "与非寄主作物轮作3年以上；及时拔除田间病株并深埋或烧毁；适期播种，避免播种过深。",
                "chemical_control": {
                    "recommendations": [
                        {"agent_name": "戊唑醇", "usage": "60克/升悬浮种衣剂按药种比1:500包衣", "precaution": "种子包衣是防治该病最有效措施之一"},
                        {"agent_name": "咯菌腈·精甲霜灵", "usage": "62.5克/升悬浮种衣剂按药种比1:300包衣", "precaution": "包衣均匀，晾干后播种"}
                    ]
                }
            },
            "prevention_tips": "选用抗病品种是防治丝黑穗病最根本的措施；避免施用带菌粪肥。"
        },
        "黑粉病": {
            "current_analysis": "检测到玉米黑粉病（又称瘤黑粉病）。该病可危害玉米地上部任何幼嫩组织，形成大小不等的肿瘤状菌瘿。菌瘿初期白色，后变为灰黑色，破裂后散出大量黑粉（冬孢子）。",
            "risk_assessment": "病菌通过伤口侵染，风雨、冰雹、虫害造成的伤口利于发病。高温干旱或干湿交替的天气条件适宜病害发生。",
            "control_measures": {
                "agricultural_control": "选用抗病品种；减少机械损伤；及时防治玉米螟等蛀茎害虫；发现菌瘿后立即摘除并深埋。",
                "chemical_control": {
                    "recommendations": [
                        {"agent_name": "戊唑醇", "usage": "430克/升悬浮剂15-20毫升/亩喷雾预防", "precaution": "抽雄前喷药保护幼嫩组织，发病后摘除菌瘿后再喷药"},
                        {"agent_name": "三唑酮", "usage": "15%可湿性粉剂60-80克/亩", "precaution": "可结合防治叶部病害一并施药"}
                    ]
                }
            },
            "prevention_tips": "减少田间操作造成的机械损伤；及时防治玉米螟，降低侵染入口。"
        },
        "穗腐病": {
            "current_analysis": "检测到玉米穗腐病。果穗顶端或基部开始发病，籽粒变色、霉烂，表面产生白色、粉红色或灰绿色霉层。病粒含多种真菌毒素，对人和牲畜健康危害极大。",
            "risk_assessment": "灌浆期至成熟期遇连续阴雨、高温高湿条件易诱发穗腐病。虫害（特别是玉米螟）造成的伤口为病菌侵入提供通道。苞叶短小、包裹不严的品种更易感病。",
            "control_measures": {
                "agricultural_control": "选用苞叶包裹紧密的抗病品种；适期早播，避开灌浆期雨季；及时防治玉米螟；收获后及时晾晒、烘干。",
                "chemical_control": {
                    "recommendations": [
                        {"agent_name": "吡唑醚菌酯", "usage": "250克/升乳油30-40毫升/亩", "precaution": "抽雄吐丝期喷药保护果穗，重点喷施穗部"},
                        {"agent_name": "戊唑醇·肟菌酯", "usage": "75%水分散粒剂15-20克/亩", "precaution": "结合防治玉米螟的杀虫剂一起喷施，事半功倍"}
                    ]
                }
            },
            "prevention_tips": "选用抗病品种；加强害虫防治；收获后剔除霉变籽粒，防止毒素污染。"
        },
        # ---------- 病毒病 ----------
        "粗缩病": {
            "current_analysis": "检测到玉米粗缩病。由水稻黑条矮缩病毒引起，灰飞虱持久性传毒。典型症状为植株严重矮化、节间缩短，叶片宽厚浓绿、僵直，叶背面有蜡白色突起条斑。感病植株多不能抽穗或抽穗极小。",
            "risk_assessment": "灰飞虱种群数量是决定发病轻重的关键因素。春季温暖干燥利于灰飞虱越冬和繁殖，套种田、早播田发病重于纯作田和晚播田。",
            "control_measures": {
                "agricultural_control": "调整播期，避开灰飞虱迁飞高峰；清除田间及周边杂草，减少毒源和虫源；拔除田间病株。",
                "chemical_control": {
                    "recommendations": [
                        {"agent_name": "吡虫啉", "usage": "10%可湿性粉剂20-30克/亩", "precaution": "玉米出苗后至7叶期前是防治灰飞虱的关键时期，每隔5-7天喷1次，连续2-3次"},
                        {"agent_name": "噻虫嗪", "usage": "25%水分散粒剂10-15克/亩", "precaution": "可同时加入氨基寡糖素等抗病毒剂提高植株抗性"}
                    ]
                }
            },
            "prevention_tips": "防治策略是“切断毒源、治虫防病”，重点控制苗期灰飞虱。选用抗耐病品种。"
        },
        "矮花叶病": {
            "current_analysis": "检测到玉米矮花叶病。由玉米矮花叶病毒引起，蚜虫非持久性传播。典型症状为叶片出现黄绿相间的条纹或斑驳，植株矮化，穗小粒少。",
            "risk_assessment": "蚜虫发生量直接影响病害流行程度。高温干旱年份蚜虫繁殖快，发病重。田间毒源杂草（如狗尾草）多的地块发病风险高。",
            "control_measures": {
                "agricultural_control": "选用抗病品种；清除田间及周边杂草；适期晚播，避开蚜虫迁飞高峰。",
                "chemical_control": {
                    "recommendations": [
                        {"agent_name": "吡蚜酮", "usage": "50%可湿性粉剂15-20克/亩", "precaution": "苗期发现蚜虫立即防治，7-10天后再防一次"},
                        {"agent_name": "高效氯氟氰菊酯", "usage": "2.5%乳油20-30毫升/亩", "precaution": "可与抗病毒剂（如香菇多糖）混用增强防效"}
                    ]
                }
            },
            "prevention_tips": "选用抗病品种是最经济有效的措施；加强田间管理，促进植株健壮生长。"
        },
        # ---------- 默认兜底 ----------
        "未知病害": {
            "current_analysis": "未能识别出具体病害类型，或检测到多种混合感染。当前症状可能不典型，或由多种病原复合侵染所致。",
            "risk_assessment": "请密切关注植株状态，防止病情蔓延。建议实地踏查，观察病害发展动态。",
            "control_measures": {
                "agricultural_control": "保持田间通风透光，及时清理病叶、病株；合理灌溉，避免田间积水；增施磷钾肥提高植株抗病力。",
                "chemical_control": {
                    "recommendations": []
                }
            },
            "prevention_tips": "建议咨询当地农技服务中心或植保专家进行实地诊断，以获得更精确的病害识别和防治方案。同时注意观察病害发生部位、病斑形态、霉层颜色等特征，便于后续精准识别。"
        }
    }

    # 返回匹配的模板，若找不到则返回默认的“未知病害”
    return templates.get(main_disease, templates["未知病害"])