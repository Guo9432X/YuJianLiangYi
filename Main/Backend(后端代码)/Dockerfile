# # 使用轻量级的 slim 版本作为基础镜像
# FROM python:3.11-slim AS builder

# WORKDIR /app

# # 复制依赖文件并安装，在这一步进行清理，减小中间层体积
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # 生产阶段，复制构建阶段的依赖和业务代码
# FROM python:3.11-slim
# WORKDIR /app

# # 从构建阶段复制已安装的依赖
# COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
# COPY --from=builder /usr/local/bin/ /usr/local/bin/

# # 复制项目代码和模型权重文件
# COPY . .

# # 配置环境变量，优化PyTorch性能
# ENV OMP_NUM_THREADS=1

# EXPOSE 80
# CMD ["gunicorn", "--bind", "0.0.0.0:80", "app:app"]
# 使用轻量级的 slim 版本作为基础镜像
# 阶段一：构建环境
# 阶段一：编译环境 (Builder)



# FROM python:3.11-slim AS builder

# WORKDIR /app

# # 1. 配置 pip 源加速下载，防止超时
# RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
#     pip config set global.extra-index-url https://download.pytorch.org/whl/cpu

# COPY requirements.txt .

# # 2. 安装依赖 (显式指定安装 cpu 版 torch)
# RUN pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu && \
#     pip install --no-cache-dir -r requirements.txt && \
#     pip install --no-cache-dir opencv-python-headless

# # 阶段二：生产环境 (Runtime)
# FROM python:3.11-slim
# WORKDIR /app

# # 3. 安装 YOLOv8 和 OpenCV 运行必需的底层系统库 (非常关键)
# RUN apt-get update && apt-get install -y \
#     libglib2.0-0 \
#     libgl1 \
#     && rm -rf /var/lib/apt/lists/*

# # 从构建阶段复制 site-packages
# COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
# COPY --from=builder /usr/local/bin/ /usr/local/bin/

# # 复制项目代码和模型
# COPY . .

# # 环境变量优化
# ENV OMP_NUM_THREADS=1
# ENV PYTHONUNBUFFERED=1

# EXPOSE 80
# # 适当增加 gunicorn 超时时间，防止模型初始化时间过长被杀掉
# CMD ["gunicorn", "--bind", "0.0.0.0:80", "--timeout", "600", "app:app"]

FROM python:3.11-slim AS builder

WORKDIR /app

# pip清华加速
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .

RUN pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir opencv-python-headless

# 生产镜像
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

COPY . .

ENV OMP_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1

EXPOSE 80

# ===== 关键：云托管启动探测超时拉长到2分钟 =====
HEALTHCHECK --start-period=120s --timeout=20s --interval=15s --retries=20 \
CMD curl -s http://127.0.0.1:80/ >/dev/null || exit 1

# gunicorn 超时开到10分钟，给YOLO模型完整加载时间
CMD ["gunicorn", "--bind", "0.0.0.0:80", "--timeout", "600", "--keep-alive", "120", "app:app"]
