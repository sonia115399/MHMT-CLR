# MHMT-CLR / MHMT-Net

论文 **MHMT-CLR: An Extensible Edge-Cloud Intelligent Diagnosis Framework for Multi-Species Leaf Disease Classification** 的整理代码。`code/` 对应 **MHMT-Net**，`SimCLR/` 对应 **MHMT-CLR**；二者共用 ResNet18 编码器和物种分支，分别使用监督学习与 SimCLR 完成番茄源域预训练。

仓库提供训练、测试、预测、论文实验配置和本地测试。数据集、预训练权重、训练结果、划分记录及论文草稿均不随仓库发布。

## 目录

```text
code/                   MHMT-Net 原始下游脚本、依赖与统一入口
SimCLR/                 MHMT-CLR 原始下游脚本、依赖与统一入口
mhmt/                   共用网络、数据增强、划分、训练与指标
configs/paper.json      预训练、物种训练设置、五个种子及论文参考数量
scripts/evaluate.py     对测试集重新评估最佳模型
scripts/predict.py      单张图像预测
tests/                  网络、协议、完整流程测试
```

## 环境

论文报告的环境为 Python 3.8、PyTorch 1.10.0、CUDA 11.3；GPU 为 NVIDIA Tesla V100-SXM3-32GB。推荐在 Linux CUDA 环境运行完整训练。先创建 Python 3.8 环境，再按 [PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/) 安装 PyTorch 1.10.0 与 CUDA 11.3，随后安装依赖：

```bash
python -m pip install -r requirements.txt
```

本地 CPU 流程检查可使用 Python 3.11 与独立的现代依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-modern.txt -r requirements-dev.txt
python -m pytest
```

现代环境用于验证实现与运行流程；论文完整实验请采用其报告的训练环境。脚本默认自动选择 CUDA 或 CPU，`--device cpu` 可指定 CPU。

## 数据准备

数据来源对应论文参考文献 [30]–[33]，数据集不随代码发布：

- 番茄：[Tomato leaf disease detection](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf)，论文源域为 10 类、每类 1,100 张，共 11,000 张。
- 玉米：[Plant Disease Detection](https://www.kaggle.com/datasets/karagwaanntreasure/plant-disease-detection)，论文为 4 类、7,316 张。
- 椰枣：[Date Palm data](https://www.kaggle.com/datasets/hadjerhamaidi/date-palm-data)，论文为 3 类、1,200 张。
- 茶叶：[Tea Sickness Dataset](https://www.kaggle.com/datasets/mexwell/tea-sickness-dataset)，论文为 8 类、885 张。

保持下游原始目录式划分：

```text
data/
  tomato/
    train/<番茄类别名>/<图像>
    test/<番茄类别名>/<图像>
  corn/
    train/<玉米类别名>/<图像>
    test/<玉米类别名>/<图像>
  datepalm/
    train/<椰枣类别名>/<图像>
    test/<椰枣类别名>/<图像>
  tea/
    train/<茶叶类别名>/<图像>
    test/<茶叶类别名>/<图像>
```

下游 `train/` 和 `test/` 必须包含相同类别目录。统一入口的类别按目录名排序，名称保存到 checkpoint，评估与预测沿用该顺序。训练与验证使用互不重叠的索引；统一入口的验证和测试使用确定性预处理。

番茄源域支持上述 `train/test` 外层目录，也支持 `tomato/<类别>/<图像>`。传入外层目录时，原有 train 与 test 图像合并为论文定义的源域预训练语料，源域目录名称不作为标签。论文源域共 10 类，每类 1,100 张；需要检查这一数量时添加 `--strict-paper-source`。

`--data-root` 也可以指向原有 `code/multi_data/corn`、`SimCLR/multi_data/tea` 等本地路径，无须上传、移动或重新划分数据。比较两种方法时，应显式传入同一份数据。

## 预训练

所有命令从仓库根目录运行。两个方法使用同一番茄源域：

```bash
# MHMT-Net：交叉熵监督预训练，100 轮
python code/run.py --data-root data/tomato --output-dir outputs/net_source

# MHMT-CLR：SimCLR / InfoNCE，1000 轮
python SimCLR/run.py --data-root data/tomato --output-dir outputs/clr_source
```

## 下游训练

为每个物种单独选择分支，重用对应方法的源域编码器：

```bash
python code/train.py --species corn --data-root data/corn --checkpoint outputs/net_source/source.pth
python code/train.py --species datepalm --data-root data/datepalm --checkpoint outputs/net_source/source.pth
python code/train.py --species tea --data-root data/tea --checkpoint outputs/net_source/source.pth

python SimCLR/train.py --species corn --data-root data/corn --checkpoint outputs/clr_source/source.pth
python SimCLR/train.py --species datepalm --data-root data/datepalm --checkpoint outputs/clr_source/source.pth
python SimCLR/train.py --species tea --data-root data/tea --checkpoint outputs/clr_source/source.pth
```

编码器参数与 BatchNorm 运行统计量固定，仅训练物种头、尾。反向传播经过编码器到达头部，因此没有在编码器前向传播中使用 `no_grad`。每次运行以验证准确率最高的 checkpoint 评估测试集，首次达到最高准确率的轮次胜出。

例如玉米 MHMT-CLR 的种子 42 输出位于：

```text
outputs/clr/corn/head1_tail1_fraction1/seed_42/
  best.pth          最佳验证模型及类别、训练设置、源模型哈希
  split.json        本次实际使用的训练/验证/测试文件列表
  history.json      各轮训练损失、验证准确率与学习率
  metrics.json      测试指标、分类报告与混淆矩阵
```

分支目录另外保存 `runs.json` 和 `summary.json`。汇总包含 top-1 accuracy、macro precision、macro recall 和 macro F1 的均值与样本标准差（`ddof=1`）。

`--seeds 42` 可只运行一次；`--epochs 1` 可缩短流程检查；`--output-dir` 指定新的结果目录，避免覆盖已有运行。这些覆盖参数对应自定义实验。

## 消融、评估和预测

表 5 的结构消融通过 `--no-head`、`--no-tail` 或同时添加二者运行。去掉 CNN 尾后保留全局池化与 512→512→C 分类 MLP，编码器仍被冻结。茶叶标注比例可用 `--fraction 0.25`、`0.5`、`0.75`、`1.0` 指定，逐类训练样本数向下取整；这些样本来自当前运行随机得到的训练划分。

```bash
python SimCLR/train.py --species tea --data-root data/tea --checkpoint outputs/clr_source/source.pth --no-head

python scripts/evaluate.py --checkpoint outputs/clr/corn/head1_tail1_fraction1/seed_42/best.pth --data-root data/corn/test

python scripts/predict.py --checkpoint outputs/clr/corn/head1_tail1_fraction1/seed_42/best.pth --image data/corn/test/Corn___Healthy/example.jpg
```

## 原始下游脚本

以下六个文件包含完整头尾模型和训练流程。从仓库根目录运行，先额外安装原始脚本依赖：

```bash
python -m pip install -r requirements-original.txt
python code/light_corn.py --help
python SimCLR/light_cornhead.py --help
```

自行提供与各文件的 encoder/尾部 state_dict 键名和尺寸匹配的历史权重，以下 checkpoint 路径为本地示例：

```bash
python code/light_corn.py --checkpoint checkpoints/net_corn.pth --data-root data/corn
python code/light_datepalmhead.py --checkpoint checkpoints/net_datepalm.pth --data-root data/datepalm
python code/light_tea.py --checkpoint checkpoints/net_tea.pth --data-root data/tea

python SimCLR/light_cornhead.py --checkpoint checkpoints/clr_corn.pth --data-root data/corn
python SimCLR/light_datepalmhead.py --checkpoint checkpoints/clr_datepalm.pth --data-root data/datepalm
python SimCLR/light_tea.py --checkpoint checkpoints/clr_tea.pth --data-root data/tea
```
