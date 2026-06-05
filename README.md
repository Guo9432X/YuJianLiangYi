“玉”见良医——玉米病害智能检测与防治小程序
项目背景
玉米是我国第一大粮食作物，常年受大斑病、灰斑病、锈病等多种病害威胁。传统人工巡检耗时耗力且依赖经验，市面上的智能识病工具往往缺乏对复杂农田环境的自适应能力，也无法将病害表象与气象、土壤等诱因关联形成决策闭环。

本项目旨在开发一款轻量化、可交互的微信小程序，以数字图像处理（DIP）、深度学习目标检测（YOLOv8） 与大模型智能体为核心技术，实现多类玉米叶片病害的自动识别、健康评估与智能防治建议生成，辅助农户和农技人员快速定位、处理和预防田间病害问题。

技术栈
层级	技术选型
前端	微信小程序原生框架 (WXML + WXSS + JavaScript)
后端	Python 3.9+、Flask / Gunicorn
深度学习	Ultralytics YOLOv8
图像处理	OpenCV、NumPy
大模型	DeepSeek API（deepseek‑chat）
外部服务	高德地图（逆地理编码）、Visual Crossing（天气API）
数据库与存储	微信云开发（云数据库、云存储）
部署	Docker、Gunicorn、微信云托管（可选）
系统架构
系统采用微信小程序 + 云开发 + 自建后端的混合架构：

小程序前端：提供病害识别、记录管理、用户中心等界面，调用后端API完成检测。

后端服务：部署于云服务器，提供 /detect 等REST接口，内部集成DIP增强、YOLO推理、ReAct智能体后处理、环境信息获取及大模型建议生成。

云开发：用于用户鉴权、历史记录存储、云函数（如登录回调）及图片云存储。

完整数据流：

text
用户拍照/选图 → 可选DIP增强 → 后端YOLO检测 → ReAct后处理 → 融合天气/位置 → DeepSeek生成建议 → 返回结构化结果 → 前端展示并存入云数据库

yujian-liangyi/
├── README.md                        # 项目总体说明文档
├── backend/                         # 后端代码目录
│   ├── app.py                       # Flask主入口
│   ├── requirements.txt             # Python依赖
│   ├── Dockerfile                   # Docker构建文件
│   ├── .dockerignore
│   ├── best.pt                      # YOLOv8模型权重
│   ├── dip_enhance.py               # DIP图像增强
│   ├── denoise.py                   # 去噪辅助
│   ├── resize_normalize.py          # 图像归一化
│   ├── agent_post_processor.py      # ReAct智能体后处理
│   ├── env_service.py               # 环境信息（天气/位置）
│   ├── suggestion_builder.py        # 大模型提示词与降级模板
│   ├── utils/                       # 通用工具函数
│   └── ...
├── miniprogram/                     # 微信小程序前端代码
│   ├── app.js                       # 小程序入口逻辑
│   ├── app.json                     # 全局配置
│   ├── app.wxss                     # 全局样式
│   ├── project.config.json          # 项目配置
│   ├── sitemap.json                 # 站点地图
│   ├── envList.js                   # 云开发环境ID列表
│   ├── images/                      # 静态图片资源
│   ├── pages/                       # 页面目录
│   │   ├── detect/                  # 病害检测页
│   │   ├── record/                  # 历史记录页
│   │   └── user/                    # 个人中心页
│   ├── utils/                       # 前端工具函数
│   └── components/                  # 自定义组件
└── cloudfunctions/                  # 微信云函数
    └── quickstartFunctions/         # 示例云函数（登录、数据库操作等）
核心模块说明（分模块负责人）
模块	文件	负责人	功能简述
DIP图像增强	dip_enhance.py	郭俊岑	光照补偿、CLAHE均衡化、病斑边缘锐化
ReAct智能体后处理	agent_post_processor.py	郭俊岑	置信度校准、小目标补偿、热力图生成
防治建议生成器	suggestion_builder.py	郭俊岑	提示词构建、DeepSeek API调用、本地降级
YOLOv8检测集成	resize_normalize.py + best.pt	周嘉明	模型加载、图像预处理、原始检测输出
环境信息服务	env_service.py	占传润	高德逆地理编码、Visual Crossing天气获取
后端API与调度	app.py	占传润	Flask路由、请求处理、模块串联
小程序前端	miniprogram/	廖崧琳、占传润	页面交互、图像上传、结果展示、云数据库操作
测试
单元测试
bash
cd backend
python -m unittest discover tests/
集成测试
启动后端服务（本地或远程）。

使用 test_local.py（由郭俊岑提供）验证三个核心模块：

bash
python test_local.py
该脚本会在 output/ 目录生成增强后的图片及热力图。

微信开发者工具中模拟真机测试完整流程。

性能指标
图像自适应增强 ≤ 1秒

YOLO检测 + ReAct后处理 ≤ 3秒

DeepSeek建议生成 10–20秒（取决于网络）

支持 ≥20 用户并发

弱光/逆光场景检测准确率衰减 ≤ 5%

人工智能大模型应用说明
使用的模型：DeepSeek API（deepseek-chat，2026年调用）

应用环节：仅在“启智模式”下生成防治建议，不参与图像检测。

输出控制：强制要求JSON格式，并进行字段完整性校验。

安全措施：对药剂用量进行规则过滤，防止超额用药；输出与植保手册逻辑一致性验证。

降级策略：API不可用时自动切换至本地预置的9种常见病害防治模板。

项目进度与分工
本项目采用敏捷开发，主要阶段及甘特图如下：

阶段	时间	主要任务	参与人员
项目启动与需求分析	2026.03.10 – 03.16	背景调研、需求规格	全体
系统设计	2026.03.17 – 03.30	架构设计、详细设计、开发规范	全体
核心功能开发	2026.03.24 – 04.20	DIP、YOLO、Agent、前端页面、后端API	分模块负责人
系统集成与联调	2026.04.07 – 04.23	模块集成、前后端联调	全体
测试与版本交付	2026.04.21 – 04.27	功能测试、兼容性测试、部署上线	全体
文档完善与答辩准备	2026.04.28 – 05.30	文档归档、系统优化、答辩材料	全体
详细甘特图见项目根目录 甘特图v2.png。

常见问题
Q：为什么不在小程序端直接运行YOLO模型？
A：模型文件较大（约80MB），受限于小程序包体积限制，且云端部署便于模型热更新和多用户并发。

Q：天气和位置信息是否必须？
A：仅在“启智模式”下需要，常规检测模式无需位置授权。

Q：如何提高早期微小病斑的检测能力？
A：建议开启“图像增强”开关，后端ReAct智能体会自动补偿符合条件的低置信度候选框。

Q：DeepSeek API Key不配置会怎样？
A：系统会完全降级为本地模板，仍能返回基础防治建议，但无法根据实时天气和位置个性化输出。

许可与贡献
本项目为2026年中国大学生计算机设计大赛参赛作品，遵循 MIT License 开源。

团队成员：

郭俊岑（DIP、Agent、大模型）

周嘉明（YOLO检测、算法集成）

廖崧琳（小程序前端、UI设计）

占传润（后端API、环境服务、云开发）

联系方式：yujianliangyi@example.com

让手机成为农民的植保专家 —— “玉”见良医，守护每一株玉米的健康。

本回答由 AI 生成，内容仅供参考，请仔细甄别。
不需要说得那么详尽，只需要将清楚这个项目本身就行了
我们要求简化，只讲清楚项目本身。需要给出一个精简的整体README，不包含过度细节。结合之前内容：项目背景、技术栈、目录结构、快速开始（可选但精简）、模块负责人（可选）、AI说明。但用户说“只需要讲清楚这个项目本身就行了”，避免冗长。我将生成一个简洁版README。

“玉”见良医——玉米病害智能检测与防治小程序
项目简介
“玉”见良医是一款基于微信小程序的玉米病害智能检测工具，为农户和农技人员提供病害识别、健康评估、防治建议一站式服务。

核心流程：用户拍摄或上传玉米叶片图像 → 可选图像增强 → YOLOv8多病害检测 → ReAct智能体后处理优化 → 融合天气与位置信息 → DeepSeek大模型生成个性化防治报告。

技术架构
前端：微信小程序原生框架（WXML + WXSS + JavaScript）

后端：Python 3.9+、Flask、Gunicorn

深度学习：YOLOv8

图像处理：OpenCV、NumPy

大模型：DeepSeek API（deepseek-chat）

外部服务：高德地图（逆地理编码）、Visual Crossing（天气）

数据与存储：微信云开发（云数据库、云存储）

目录结构
text
yujian-liangyi/
├── backend/                     # 后端代码
│   ├── app.py                   # Flask主入口
│   ├── best.pt                  # YOLOv8模型权重
│   ├── dip_enhance.py           # 图像增强（光照/病斑）
│   ├── agent_post_processor.py  # ReAct智能体后处理
│   ├── suggestion_builder.py    # 大模型提示词与降级模板
│   ├── env_service.py           # 天气/位置获取
│   └── requirements.txt
├── miniprogram/                 # 小程序前端代码
│   ├── pages/                   # 检测页、记录页、我的页
│   ├── utils/                   # 工具函数
│   ├── images/                  # 静态资源
│   └── app.js / app.json / app.wxss
├── cloudfunctions/              # 微信云函数（登录、数据库）
└── docs/                        # 项目文档（需求、设计、测试报告等）


核心模块说明（分模块负责人）
模块	文件	负责人	功能简述
DIP图像增强	dip_enhance.py	郭俊岑	光照补偿、CLAHE均衡化、病斑边缘锐化
ReAct智能体后处理	agent_post_processor.py	郭俊岑	置信度校准、小目标补偿、热力图生成
防治建议生成器	suggestion_builder.py	郭俊岑	提示词构建、DeepSeek API调用、本地降级
YOLOv8检测集成	resize_normalize.py + best.pt	周嘉明	模型加载、图像预处理、原始检测输出
环境信息服务	env_service.py	占传润	高德逆地理编码、Visual Crossing天气获取
后端API与调度	app.py	占传润	Flask路由、请求处理、模块串联
小程序前端	miniprogram/	廖崧琳、占传润	页面交互、图像上传、结果展示、云数据库操作
