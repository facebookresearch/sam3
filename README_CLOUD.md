# SAM3 训练与推理代码

这份目录只包含在云服务器上训练和推理要用的代码。权重、数据集、训练输出和笔记本都没有放进来，方便直接推到 GitHub 再 `git clone`。

克隆之后把本目录当成仓库根目录使用。

## 包含什么

- `sam3/`：模型、训练、评估。文本编码用的 BPE 词表在 `sam3/assets/bpe_simple_vocab_16e6.txt.gz`。
- `examples/`：图像推理、把训练 checkpoint 合并成可加载权重、RealSense 实时推理。
- `ONNX/`：导出 ONNX 的脚本。导出的 `.onnx` 不在这里。
- `pyproject.toml`：安装入口。训练说明见 `README_TRAIN.md`。

## 不包含什么

- `sam3.pt`、微调得到的 `*.pt`、checkpoint
- `data/`、`Tank.coco-segmentation*`、`Factory_photo/`
- `outputs/`、`output/`
- 笔记本和演示图片、视频

这些文件用别的方式传到服务器，不要提交到 Git。`.gitignore` 已经忽略 `*.pt`、`*.pth`、`*.onnx`、`*.ckpt`。

## 安装

```bash
conda create -n sam3 python=3.12
conda activate sam3
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[train]"
pip install opencv-python pycocotools pillow
```

CUDA 版本和本机驱动不一致时，换成对应的 PyTorch 安装命令。

## 训练前要改的路径

`sam3/train/configs/tank/` 里的路径还是本机绝对路径，克隆到云服务器后改成那台机器上的位置：

- `paths.roboflow_vl_100_root`：COCO 数据根目录。Tank 配置期望下面有 `tank/train`、`tank/test`，或 shelf 配置里的 `tank_shelf/`。
- `paths.experiment_log_dir`：日志和 checkpoint 输出目录。
- `paths.bpe_path`：指向本仓库里的 `sam3/assets/bpe_simple_vocab_16e6.txt.gz`。
- `trainer.model.checkpoint_path`：初始权重。第一轮微调指向官方 `sam3.pt`，后续配置指向上一轮合并好的 `sam3_tank_head.pt`。

官方 `sam3.pt` 需要在 Hugging Face 申请权限后自行下载，不要放进 Git。

## 训练

在本目录下执行：

```bash
python sam3/train/train.py -c configs/tank/tank_head_ft.yaml --use-cluster 0 --num-gpus 1
```

Shelf 数据用 `configs/tank/tank_head_shelf.yaml`。配置里的路径是相对 `sam3/train/` 解析的。

## 推理

训练保存的是 head checkpoint，先合并进完整的 `sam3.pt`：

```bash
python examples/export_sam3_head_checkpoint.py \
  --base /path/to/sam3.pt \
  --train-ckpt /path/to/outputs/tank_head_ft/checkpoints/checkpoint.pt \
  --output /path/to/sam3_tank_head.pt
```

再对一个图片目录做分割：

```bash
python examples/segment_factory_photos.py \
  --image-dir /path/to/images \
  --checkpoint /path/to/sam3_tank_head.pt \
  --prompt "Tank body,Tank handle" \
  --output-dir /path/to/output
```
