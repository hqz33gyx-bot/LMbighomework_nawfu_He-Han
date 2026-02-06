# 基于AC-RF的Z-Image-Turbo模型微调与量化技术
## 摘要
本研究针对Z-Image文生图模型的高分辨率生成与极速推理需求，提出基于**锚点约束整流流（Anchor-Constrained Rectified Flow, AC-RF）**的微调方案。模型采用扩散Transformer（DiT）架构，结合低秩适应（LoRA）技术实现特定领域（时尚单品）的高效适配；通过AC-RF算法优化生成轨迹，配合多损失函数约束与显存优化策略，实现4-8步的极速推理（Turbo Inference），同时保证生成图像的纹理与结构保真度。

## 1. Z-Image 模型架构
本系统核心模型基于**可扩展的单流扩散 Transformer（Scalable Single-Stream Diffusion Transformer, DiT）**架构，相较于传统U-Net架构，DiT在高分辨率图像生成任务中具备更优的长程依赖建模能力。模型架构的核心设计如下：
1.  **空间感知增强**

    引入**旋转位置编码（Rotary Positional Embeddings, RoPE）**，提升Transformer对图像空间位置信息的捕捉能力。
2.  **条件注入机制**

    通过**自适应层归一化（AdaLN-Zero）**模块，将时间步（Timestep）与文本条件信息注入至每个Transformer块，实现文本与图像的精准对齐。
3.  **高效微调策略**

    采用**低秩适应（Low-Rank Adaptation, LoRA）**技术，在保留预训练模型通用先验的前提下完成领域适配。LoRA模块嵌入至Transformer注意力层（`to_q`/`to_k`/`to_v`/`to_out`）与前馈网络层（`w1`/`w2`/`w3`）的线性投影层，大幅降低可训练参数量，实现时尚单品领域的高效微调。

## 2. 锚点约束整流流（AC-RF）加速算法
为解决传统扩散模型推理步数多、生成速度慢的痛点，本研究引入**锚点约束整流流（AC-RF）**算法，对Z-Image模型进行Turbo加速微调。AC-RF是改进型流匹配（Flow Matching）方法，核心思想是在噪声分布与数据分布之间构建**笔直化传输轨迹**，通过以下关键技术组件实现极速推理：
1.  **混合流匹配损失**

    结合L1损失与余弦相似度损失，同时优化速度场的幅值与方向，提升流匹配的精准度。
2.  **频率感知损失**

    针对时尚单品纹理复杂的特性，通过傅里叶变换将图像分解为高频（布料纹理）与低频（服装结构）分量，赋予不同权重（$\omega_{high}$/$\omega_{low}$），强化模型对不同频率特征的学习能力。
3.  **风格-结构解耦损失**

    基于VAE潜在空间特征，分别计算**风格损失（Gram Matrix）**与**结构特征损失**，约束模型在微调过程中保持生成图像的几何一致性与材质真实感。
4.  **曲率惩罚**

    在训练后期引入曲率惩罚项，通过最小化生成轨迹的二阶导数，鼓励模型学习“匀速直线”轨迹，最终支持**4-8步的极速推理**。
5.  **锚点机制与采样策略**

    训练过程中引入**锚点（Anchors）**机制，结合重放缓冲区（Replay Buffer）与定制化采样策略，强制生成轨迹经过预定义锚点时间步，进一步压缩推理步数。

## 3. 实验设置
### 3.1 实验数据集：Fashion-Gen
微调实验采用**Fashion-Gen**数据集，该数据集由Rostamzadeh等人发布，具备以下特性：
- 数据规模：包含293,008张高清晰度时尚单品图像，原图分辨率为1360×1360。
- 标注质量：每张图像配有专业造型师撰写的详细文本描述，涵盖面料、剪裁、纹理等关键属性。
- 预处理流程：将图像统一调整为**[待补充]×[待补充]**分辨率；清洗文本描述，去除无关特殊字符。
- 数据多样性：涵盖48个主类别（夹克、衬衫、毛衣等）与121个细分类别；每个单品包含1-6个不同视角图像，且具备一致的影棚光照背景，是高保真虚拟资产生成模型的理想训练数据。

### 3.2 训练参数与策略
实验基于Ubuntu 20.04系统与NVIDIA GPU硬件平台，采用`accelerate`库实现分布式训练，通过BF16混合精度训练策略节省显存并提升计算效率。核心训练超参数与优化策略如下：
| 类别 | 参数配置 |
|------|----------|
| 基础模型 | Z-Image-Turbo（DiT架构） |
| 优化器 | AdamW8bit，，权重衰减=0 |
| 学习率调度 | 余弦退火重启策略（Cosine with Restarts），预热步数=100 |
| LoRA配置 | Rank=16，Alpha=16，作用于所有Transformer块线性层 |
| 时间步采样 | Shifted Sigmoid采样（Shift=3.0），聚焦中间信噪比区域 |
| 损失函数权重 | L1 Loss权重；频率感知损失权重；风格结构损失权重 |
| 梯度优化 | Min-SNR-Gamma加权策略（$\gamma$），平衡不同噪声水平样本的梯度贡献 |
| 显存优化 | 采用Block Swapping技术与Gradient Checkpointing策略|


## 4. 快速开始
### 4.1 环境准备
```bash
# 进入项目目录
cd Z-image-Turbo_trainer

# 一键安装依赖
chmod +x setup.sh
./setup.sh

# 编辑配置文件
cp env.example .env
nano .env
```

### 4.2 启动 Web UI 服务
```bash
# 启动服务
./start.sh

# 部署完成后，浏览器访问
# http://localhost:9198
```

## 5. 命令行使用指南
### 5.1 生成缓存
#### 5.1.1 生成 Latent 缓存（VAE 编码）
```bash
python -m zimage_trainer.cache_latents \
--vae Z-Image-Turbo/vae \
--input_dir datasets/fashion-dataset \
--output_dir datasets/fashion-dataset \
--batch_size 4
```

#### 5.1.2 生成 Text 缓存（文本编码）
```bash
python -m zimage_trainer.cache_text_encoder \
--text_encoder Z-Image-Turbo/text_encoder \
--input_dir datasets/fashion-dataset \
--output_dir datasets/fashion-dataset \
--max_length 512
```

### 5.2 启动训练
```bash
python -m accelerate.commands.launch --mixed_precision bf16 \
    scripts/train_zimage_v2.py --config config/fashion-dataset.toml
```

### 5.3 推理生成
```bash
python -m zimage_trainer.inference \
    --model_path ./Z-Image-Turbo \
    --lora_path ./output/*.safetensors \
    --prompt "xxx" \
    --output_path ./output/generated.png \
    --num_inference_steps 10
```

## 6. 训练参数详解
配置文件路径：`config/acrf_config.toml`

```toml
[acrf]
turbo_steps = 10        # 锚点数（推理步数）
shift = 3.0             # Z-Image 官方值
jitter_scale = 0.02     # 锚点抖动

[lora]
network_dim = 16        # LoRA rank
network_alpha = 16      # LoRA alpha

[training]
learning_rate = 1e-4    # 学习率
num_train_epochs = 10   # 训练轮数
snr_gamma = 5.0         # Min-SNR 加权
loss_mode = "standard"  # 损失模式（见下方说明）

[dataset]
batch_size = 1
enable_bucket = true
max_sequence_length = 512  # 文本序列长度
```

### 6.1 损失模式总览
| 模式 | 说明 | 适用场景 | 推荐参数 |
|------|------|----------|----------|
| standard | 基础 MSE + 可选 FFT/Cosine | 通用训练 | 默认即可 |
| frequency | 频域感知（高频L1 + 低频Cosine） | 锐化细节，不改风格 | alpha_hf=1.0, beta_lf=0.2 |
| style | 风格结构（SSIM + Lab统计量） | 学习大师光影/调色 | lambda_struct=1.0 |
| unified | 频域 + 风格 组合 | 全面增强 | 两者默认值 |

### 6.2 频域感知模式（frequency）
#### 6.2.1 核心原理
```
┌─────────────────────────────────────────┐
│  Latent ──► 降采样 ──► 低频（结构）     │
│         └──► 高频 = 原始 - 低频（细节） │
│                                         │
│  Loss = MSE + α·L1(高频) + β·Cos(低频)  │
└─────────────────────────────────────────┘
```
#### 6.2.2 参数说明
| 参数 | 默认值 | 作用 | 推荐范围 |
|------|--------|------|----------|
| `alpha_hf` | 1.0 | 高频（纹理/边缘）增强强度 | 0.5 ~ 1.5 |
| `beta_lf` | 0.2 | 低频（结构/光影）方向锁定 | 0.1 ~ 0.5 |

#### 6.2.3 场景配置推荐
| 场景 | alpha_hf | beta_lf | 说明 |
|------|----------|---------|------|
| 锐化细节 | 1.0~1.5 | 0.1 | 重点学习纹理 |
| 保持结构 | 0.5 | 0.3~0.5 | 防止构图偏移 |
| 平衡 | 0.8 | 0.2 | 推荐默认 |

### 6.3 风格结构模式（style）
#### 6.3.1 核心原理
```
┌─────────────────────────────────────────────┐
│  Latent 近似 Lab 空间                        │
│  ├─ L通道 ──► SSIM（锁结构）                │
│  │         ├─ Mean/Std（学光影）            │
│  │         └─ 高频L1（学纹理）              │
│  └─ ab通道 ──► Mean/Std（学色调）           │
└─────────────────────────────────────────────┘
```

#### 6.3.2 参数说明
| 参数 | 默认值 | 作用 | 推荐范围 |
|------|--------|------|----------|
| `lambda_struct` | 1.0 | SSIM 结构锁（防脸崩） | 0.5 ~ 1.5 |
| `lambda_light` | 0.5 | L 通道统计量（学光影曲线） | 0.3 ~ 1.0 |
| `lambda_color` | 0.3 | ab 通道统计量（学色调偏好） | 0.1 ~ 0.5 |
| `lambda_tex` | 0.5 | 高频 L1（质感增强） | 0.3 ~ 0.8 |

#### 6.3.3 场景配置推荐
| 场景 | struct | light | color | tex | 说明 |
|------|--------|-------|-------|-----|------|
| 人像训练 | 1.5 | 0.3 | 0.2 | 0.3 | 强锁结构防脸崩 |
| 风格迁移 | 0.5 | 0.8 | 0.5 | 0.3 | 重点学光影色调 |
| 细节增强 | 0.8 | 0.3 | 0.2 | 0.8 | 锐化纹理 |
| 平衡 | 1.0 | 0.5 | 0.3 | 0.5 | 推荐默认 |
```