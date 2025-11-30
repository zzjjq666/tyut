# TYUT - Deepfake Detection Project

## 作者
薛雅丽、张佳琦、曹晓峰

## 项目简介
本项目针对AIGC伪造图像检测任务，提出了一种基于CLIP:ViT-L/14的端到端检测算法。通过SRFormer图像增强、监督对比学习和SAM优化器等关键技术，显著提升了模型对微小伪造特征的识别能力和泛化性能。
<img width="1935" height="548" alt="ad7b2830cd61b23533f8eafb0f1df60a" src="https://github.com/user-attachments/assets/4dca7501-3ee0-4a59-a5bc-5e8417bacfda" />

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
│   └── ...
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

## 📥 数据下载与划分
数据集下载
本算法使用论文《Towards Universal Fake Image Detectors that Generalize Across Generative Models》官方提供的训练数据集。

数据集大小: 约72GB

下载方式: 从论文官方链接下载

数据格式: 图像文件 + 标注信息

数据划分说明
text
datasets/
├── train/     # 训练集 - 用于模型训练
├── val/       # 验证集 - 用于超参数调优
└── test/      # 测试集 - 用于最终性能评估
划分比例: 训练集70%，验证集15%，测试集15%

⚙️ 环境配置
硬件要求
GPU: NVIDIA GeForce RTX 4090 (或同等算力显卡)

显存: ≥ 24GB

内存: ≥ 32GB

软件环境
bash
# 创建conda环境
conda create -n deepfake_detect python=3.8
conda activate deepfake_detect

# 安装PyTorch (CUDA 12.0)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装项目依赖
pip install -r requirements.txt
requirements.txt
text
tqdm>=4.64.0
numpy>=1.21.0
opencv-python>=4.5.0
Pillow>=8.3.0
scikit-learn>=1.0.0
tensorboard>=2.7.0
albumentations>=1.0.0
einops>=0.4.0
timm>=0.5.0
omegaconf>=2.1.0
🚀 模型训练与测试
模型训练
bash
python train.py \
    --name=clip_vitl14 \
    --wang2020_data_path=datasets/ \
    --data_mode=wang2020 \
    --arch=CLIP:ViT-L/14 \
    --fix_backbone
关键参数说明:

--name: 实验名称，用于创建保存目录

--wang2020_data_path: 数据集路径

--arch: 模型架构，使用CLIP ViT-L/14

--fix_backbone: 冻结主干网络参数

模型验证
bash
python validate.py \
    --arch=CLIP:ViT-L/14 \
    --ckpt=checkpoints/clip_vitl14/model_epoch_best.pth \
    --result_folder=your_result_folder
参数说明:

--ckpt: 训练好的模型检查点路径

--result_folder: 结果保存目录

性能指标
训练时长: ~48小时 (RTX 4090)

推理速度: ~15ms/图像

峰值显存: 18GB

🔄 复现流程
完整复现步骤
环境准备

bash
conda create -n deepfake_detect python=3.8
conda activate deepfake_detect
pip install -r requirements.txt
数据准备

bash
# 下载数据集到datasets目录
# 确保目录结构正确
模型训练

bash
python train.py --name=clip_vitl14 --wang2020_data_path=datasets/ --data_mode=wang2020 --arch=CLIP:ViT-L/14 --fix_backbone
模型测试

bash
python validate.py --arch=CLIP:ViT-L/14 --ckpt=checkpoints/clip_vitl14/model_epoch_best.pth --result_folder=your_result_folder
快速验证
bash
# 使用预训练模型快速测试
python validate.py --ckpt=pretrained_models/best_model.pth --data_dir=datasets/test
⚙️ 环境配置
实验配置
配置项	规格
GPU型号	NVIDIA GeForce RTX 4090
显存	24 GB
CUDA Version	≥ 12.0
GPU驱动版本	NVIDIA 575.57.08
依赖安装
bash
# 创建conda环境
conda create -n deepfake_detect python=3.8
conda activate deepfake_detect

# 安装PyTorch (CUDA 12.0)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装项目依赖
pip install -r requirements.txt
requirements.txt 内容
text
tqdm>=4.64.0
numpy>=1.21.0
opencv-python>=4.5.0
Pillow>=8.3.0
scikit-learn>=1.0.0
tensorboard>=2.7.0
albumentations>=1.0.0
einops>=0.4.0
timm>=0.5.0
omegaconf>=2.1.0
🚀 模型训练与测试
模型训练
bash
python train.py \
    --name=clip_vitl14 \
    --wang2020_data_path=datasets/ \
    --data_mode=wang2020 \
    --arch=CLIP:ViT-L/14 \
    --fix_backbone
模型验证
bash
python validate.py \
    --arch=CLIP:ViT-L/14 \
    --ckpt=checkpoints/clip_vitl14/model_epoch_best.pth \
    --result_folder=your_result_folder
性能指标
完整训练时长: 约48小时（在RTX 4090上）

单次推理耗时: 约15ms/图像

峰值显存占用: 18GB

