“玉”见良医 - 郭俊岑负责模块本地代码说明文档

1. 模块概览

本代码包包含由郭俊岑负责开发的三个核心Python模块：
- DIP图像增强 (dip_enhance.py)**: 用于增强玉米病斑特征。
- YOLO后处理Agent (agent_post_processor.py)**: 用于优化YOLO模型的原始输出。
- 防治建议生成器 (suggestion_builder.py)**: 用于构建Prompt并调用大模型API。

2. 本地运行与测试指南

2.1环境准备:
    Python >= 3.8
    安装依赖: `pip install numpy opencv-python Pillow requests python-dotenv`

2.2配置API Key:
    在项目根目录下创建 `.env` 文件。
    在文件中添加一行: `DEEPSEEK_API_KEY="你的真实KEY"`

2.3运行测试:
    在根目录下执行 `python test_local.py`。
    脚本会自动测试所有三个模块，并在 `output/` 文件夹下生成结果图片。

3. 模块详解与对接说明
	3.1 `dip_enhance.py`
	-功能:输入BGR格式图片，输出增强后的BGR格式图片。
	-对接:由后端(占传润)在调用YOLO模型前选择性调用。
	-可调参数:HSV颜色阈值和形态学核大小，已根据测试图初步优化。

	3.2 `agent_post_processor.py`
	-功能: 输入YOLO原始输出JSON和图片尺寸，输出优化后的JSON和热力图URL。
	-对接: 由后端(占传润)在获得YOLO结果后调用。
	-关键依赖: 强依赖YOLO(周嘉明)的输出格式为 `[{"box": [...], "class": "...", "confidence": ...}]`。
	-可调参数: 内部的规则比例值，已初步估算。

	3.3 `suggestion_builder.py`
	-功能: 输入一个包含检测结果、环境信息等的 `context_data` 字典，输出结构化的防治建议JSON。
	-对接: 由后端(占传润)在完成所有信息收集后调用。
	-核心逻辑: 动态构建高质量Prompt，调用Deepseek API。
	-降级机制: API调用失败时，会自动返回基于 `_get_fallback_suggestion` 的通用模板建议。

4. 给后端(占传润)的集成步骤
	1.  将 `my_code/` 文件夹下的所有 `.py` 文件放入你的云函数Python依赖目录。
	2.  在云函数配置中，**必须**添加环境变量 `DEEPSEEK_API_KEY`。
	3.  你的主逻辑 `index.js` 中按需导入和调用这几个模块的函数。具体调用顺序和参数请参考 `test_local.py` 中的示例。
	4.  热力图的保存逻辑在 `utils.py` 中，目前是保存到本地，你需要修改它，实现上传到微信云存储并返回 `cloud://` URL。