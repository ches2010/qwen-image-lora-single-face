# Qwen-Image LoRA 单张人脸训练项目

这是一个专门用于在单张人脸图像上训练 Qwen-Image LoRA 模型的项目，集成了数据增强、自动过拟合检测和定期评估功能。

## 项目结构

qwen-image-lora-single-face/
├── README.md
├── requirements.txt
├── train.py
├── concept_prompt.txt
└── dataset/
    └── example.jpg (示例图像)

## 功能特性

- **单张图像训练**：即使只有一张人脸图像也能训练 LoRA
- **数据增强**：自动对输入图像进行增强以增加训练样本多样性
- **自动过拟合检测**：使用 SSIM 指标监控过拟合并自动停止训练
- **定期评估**：每隔指定步数生成测试图像
- **LoRA 优化**：使用低秩适应技术，高效微调大模型

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/ches010/qwen-image-lora-single-face.git
cd qwen-image-lora-single-face
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 准备数据

1. 将你的单张人脸图像放入 `dataset/` 文件夹
2. 编辑 `concept_prompt.txt` 文件，写入触发词和描述（如 `sks 人物`）

### 4. 开始训练

```bash
python train.py --instance_image_path dataset/your_image.jpg --concept_prompt_file concept_prompt.txt --output_dir output/lora_weights
```

## 配置参数

- `--instance_image_path`: 训练图像路径
- `--concept_prompt_file`: 提示词文件路径
- `--resolution`: 图像分辨率（默认1024）
- `--max_train_steps`: 最大训练步数（默认600）
- `--rank`: LoRA 秩（默认16）
- `--ssim_threshold`: 过拟合检测阈值（默认0.95）
- `--early_stopping_patience`: 早停耐心值（默认3）

## 输出文件

- `eval_images/`: 评估图像
- `checkpoint-*`: 训练检查点
- `pytorch_lora_weights.safetensors`: 最终 LoRA 权重

## 使用建议

1. **触发词选择**：使用不常见的单词作为触发词（如 sks, vxx）
2. **图像质量**：确保训练图像清晰、光线均匀
3. **参数调整**：根据训练效果调整 `ssim_threshold` 和 `rank` 参数

## 许可证

MIT License

---

*注意：此项目需要足够的 GPU 内存来运行 Qwen-Image 模型，建议使用具有至少 24GB 显存的 GPU。*
