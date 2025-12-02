# TYUT - Deepfake Detection Project

## 作者
薛雅丽、张佳琦、曹晓峰

## 项目简介
本项目针对AIGC伪造图像检测任务，提出了一种基于CLIP:ViT-L/14的端到端检测算法。通过SRFormer图像增强、监督对比学习和SAM优化器等关键技术，显著提升了模型对微小伪造特征的识别能力和泛化性能。
<img width="2074" height="509" alt="cb26282ea7fff3d4007ced3588e2b635" src="https://github.com/user-attachments/assets/dda684b1-11c6-4ff6-8e49-79ac5d5c22f3" />

---

## 🎯 算法简介

本算法以CLIP:ViT-L/14为核心特征提取器，采用端到端流程实现伪造图像检测。核心流程包括：
1. **图像预处理**：输入图像经SRFormer进行图像预增强，重构高频部分与结构信息
2. **数据增强**：采用随机高斯模糊、JPEG压缩等多维数据增强处理
3. **特征提取**：通过CLIP:ViT-L/14提取768维高层语义特征
4. **双分支设计**：
   - 分类头输出分类预测分数
   - 投影头映射为128维对比特征
5. **联合优化**：分类损失与对比学习损失联合构建总损失，使用SAM优化器进行参数更新

---

## 📂 项目结构

```text
aigc_fake_detect/
├── checkpoints/                 # 模型检查点保存目录
│   └── clip_vitl14/            # CLIP ViT-L/14训练结果
├── data/                       # 原始数据目录
├── datasets/                   # 处理后的数据集目录
│   ├── test/                  # 测试集数据
│   ├── train/                 # 训练集数据  
│   └── val/                   # 验证集数据
├── models/                     # 模型架构定义
│   ├── clip/                  # CLIP模型相关
│   ├── __init__.py
│   ├── clip_models.py         # CLIP模型实现
│   ├── imagenet_models.py     # ImageNet预训练模型
│   ├── resnet.py              # ResNet架构
│   ├── vgg.py                 # VGG架构
│   ├── vision_transformer.py  # Vision Transformer架构
├── networks/                   # 网络组件和训练逻辑
│   ├── __init__.py
│   ├── base_model.py          # 基础模型类
│   ├── contrastive_loss.py    # 对比损失函数
│   ├── sam.py                 # SAM模型集成
│   └── trainer.py             # 训练器类
├── options/                    # 配置文件目录
├── your_result_folder/        # 结果保存目录
├── train.py                   # 训练脚本
├── validate.py                # 验证脚本
├── requirements.txt           # 环境依赖文件
└── README.md
```
## 📥 数据下载与划分
数据集下载
本算法使用论文《Towards Universal Fake Image Detectors that Generalize Across Generative Models》官方提供的训练数据集。
基线数据集下载地址：https://github.com/peterwang512/CNNDetection
基线数据集大小: 约72GB

除了基线数据集以外，我们还提供了更为丰富的数据集。为了方便大家提升模型的性能，我们也将这些数据集整理了出来，方便大家进行下载训练。
AI Generated Images vs Real Images：
https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images

CIFAKE: Real and AI-Generated Synthetic Images：
https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

我们对于数据划分说明：
```text
datasets/
├── train/     # 训练集 - 用于模型训练
├── val/       # 验证集 - 用于超参数调优
└── test/      # 测试集 - 用于最终性能评估
```
划分比例: 训练集80%，验证集20%，测试集20%

⚙️ 环境配置
硬件要求
GPU: NVIDIA GeForce RTX 4090 (或同等算力显卡)

显存: ≥ 24GB

内存: ≥ 32GB

软件环境
# 创建conda环境
```text
conda create -n deepfake_detect python=3.8
conda activate deepfake_detect
```

# 安装PyTorch (CUDA 12.0)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装项目依赖
```text
absl-py==2.3.1
addict==2.4.0
basicsr==1.4.2
cachetools==5.5.2
certifi==2025.10.5
charset-normalizer==3.4.4
clip-anytorch==2.6.0
cmake==4.1.2
filelock==3.16.1
fsspec==2025.3.0
ftfy==6.2.3
future==1.0.0
google-auth==2.43.0
google-auth-oauthlib==1.0.0
grpcio==1.70.0
idna==3.11
imageio==2.35.1
importlib-metadata==8.5.0
jinja2==3.1.6
joblib==1.4.2
lazy-loader==0.4
lit==18.1.8
lmdb==1.7.5
markdown==3.7
markupsafe==2.1.5
mpmath==1.3.0
networkx==3.1
numpy==1.24.4
nvidia-cublas-cu12==12.1.3.1
nvidia-cuda-cupti-cu12==12.1.105
nvidia-cuda-nvrtc-cu12==12.1.105
nvidia-cuda-runtime-cu12==12.1.105
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu12==11.0.2.54
nvidia-curand-cu12==10.3.2.106
nvidia-cusolver-cu12==11.4.5.107
nvidia-cusparse-cu12==12.1.0.106
nvidia-nccl-cu12==2.20.5
nvidia-nvjitlink-cu12==12.9.86
nvidia-nvtx-cu12==12.1.105
oauthlib==3.3.1
opencv-python==4.12.0.88
packaging==25.0
pillow==10.4.0
platformdirs==4.3.6
protobuf==5.29.5
pyasn1==0.6.1
pyasn1-modules==0.4.2
pywavelets==1.4.1
pyyaml==6.0.3
regex==2024.11.6
requests==2.32.4
requests-oauthlib==2.0.0
rsa==4.9.1
scikit-image==0.21.0
scikit-learn==1.3.2
scipy==1.10.1
sympy==1.13.3
tb-nightly==2.14.0a20230808
tensorboard-data-server==0.7.2
tensorboardx==2.6.2.2
threadpoolctl==3.5.0
tifffile==2023.7.10
tomli==2.3.0
torch==2.0.0+cu118
torchaudio==2.4.1+cu118
torchvision==0.15.1+cu118
tqdm==4.67.1
triton==2.0.0
typing-extensions==4.13.2
urllib3==2.2.3
wcwidth==0.2.14
werkzeug==3.0.6
yapf==0.43.0
zipp==3.20.2
```
🚀 模型训练与测试
```text
模型训练
python train.py \
    --name=clip_vitl14 \
    --wang2020_data_path=datasets/ \
    --data_mode=wang2020 \
    --arch=CLIP:ViT-L/14 \
    --fix_backbone
```
关键参数说明:
```text
--name: 实验名称，用于创建保存目录

--wang2020_data_path: 数据集路径

--arch: 模型架构，使用CLIP ViT-L/14

--fix_backbone: 冻结主干网络参数
```
模型验证
```text
python validate.py \
    --arch=CLIP:ViT-L/14 \
    --ckpt=checkpoints/clip_vitl14/model_epoch_best.pth \
    --result_folder=your_result_folder
```
参数说明:

--ckpt: 训练好的模型检查点路径

--result_folder: 结果保存目录

性能指标
训练时长: ~48小时 (RTX 4090)

推理速度: ~15ms/图像

峰值显存: 18GB

🔄 复现流程
完整复现步骤：
环境准备
```text
conda create -n deepfake_detect python=3.8
conda activate deepfake_detect
pip install -r requirements.txt
```
数据准备：
下载数据集到datasets目录
# 确保目录结构正确
模型训练：
python train.py --name=clip_vitl14 --wang2020_data_path=datasets/ --data_mode=wang2020 --arch=CLIP:ViT-L/14 --fix_backbone

模型测试：
python validate.py --arch=CLIP:ViT-L/14 --ckpt=checkpoints/clip_vitl14/model_epoch_best.pth --result_folder=your_result_folder

我们还提供快速验证的方法：
# 使用预训练模型快速测试
python validate.py --ckpt=pretrained_models/best_model.pth --data_dir=datasets/test

⚙️ 环境配置
```text
实验要求：
GPU型号	NVIDIA GeForce RTX 4090
显存	24 GB
CUDA Version	≥ 12.0
GPU驱动版本	NVIDIA 575.57.08
```
依赖安装
# 创建conda环境
```text
conda create -n deepfake_detect python=3.8
conda activate deepfake_detect
```
# 安装PyTorch (CUDA 12.0)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装项目依赖
```text
pip install -r requirements.txt
requirements.txt 具体内容见前面
```
🚀 模型训练与测试
性能指标
完整训练时长: 约48小时（在RTX 4090上）


