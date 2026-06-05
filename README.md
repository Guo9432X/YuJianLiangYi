# “玉”见良医——玉米病害智能检测与防治小程序

## 项目背景

玉米是我国第一大粮食作物，常年受大斑病、灰斑病、锈病等病害威胁。传统人工巡检效率低且依赖经验，现有工具缺乏自适应能力与决策闭环。本项目以DIP、YOLOv8与大模型智能体为核心，实现病害自动识别、健康评估与防治建议生成。

## 技术栈

| 层级 | 技术选型 |
|------|----------|
| 前端 | 微信小程序原生框架 |
| 后端 | Python 3.9+、Flask / Gunicorn |
| 深度学习 | YOLOv8 |
| 图像处理 | OpenCV、NumPy |
| 大模型 | DeepSeek API |
| 外部服务 | 高德地图、Visual Crossing天气API |
| 存储 | 微信云开发（云数据库、云存储） |

## 系统架构

系统采用微信小程序 + 云开发 + 自建后端混合架构。小程序前端调用后端`/detect`接口；后端集成DIP增强、YOLO推理、ReAct后处理、环境信息获取及大模型建议生成；云开发用于用户鉴权、历史记录存储与图片存储。

**数据流**：拍照/选图 → 可选DIP增强 → YOLO检测 → ReAct后处理 → 融合天气/位置 → DeepSeek生成建议 → 前端展示并存入云数据库。

## 目录结构

```
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
│   │   ├── index/
│   │   ├── log/
│   │   ├── mine/
│   │   └── ...
│   ├── utils/                       # 前端工具函数
│   └── components/                  # 自定义组件
└── cloudfunctions/                  # 微信云函数
    └── quickstartFunctions/         # 示例云函数（登录、数据库操作等）
```

## 核心模块与负责人

| 模块 | 文件 | 负责人 | 功能 |
|------|------|--------|------|
| DIP图像增强 | dip_enhance.py | 郭俊岑 | 光照补偿、CLAHE、病斑锐化 |
| ReAct后处理 | agent_post_processor.py | 郭俊岑 | 置信度校准、小目标补偿、热力图 |
| 防治建议生成 | suggestion_builder.py | 郭俊岑 | 提示词构建、API调用、本地降级 |
| YOLO检测集成 | resize_normalize.py + best.pt | 周嘉明 | 模型加载、预处理、检测输出 |
| 环境信息服务 | env_service.py | 占传润 | 逆地理编码、天气获取 |
| 后端API与调度 | app.py | 占传润 | 路由、请求处理、模块串联 |
| 小程序前端 | miniprogram/ | 廖崧琳、占传润 | 页面交互、上传、展示、云数据库 |
