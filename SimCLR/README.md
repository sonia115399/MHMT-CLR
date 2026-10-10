# MHMT-CLR

本目录对应 MHMT-CLR，同时提供作者原始下游代码与按论文附录整理的统一入口。

完整头尾训练代码为 `light_cornhead.py`、`light_datepalmhead.py`、`light_tea.py`，已按手稿固定编码器参数与 BN、调整椰枣 SE 位置并启用其 StepLR；原始尾部训练代码为 `corn_cnn.py`、`datepalm_cnn.py`、`tea_cnn.py`。依赖位于 `models/`、`dataprocess/`、`data_aug/`、`util/`、`exceptions/`。完整网络类和训练流程保留在文件内，具体运行方式见 [原始下游脚本](../README.md#原始下游脚本)。

下列命令为统一入口，共用实现位于 `../mhmt/`，采用番茄源域 SimCLR 预训练与冻结编码器的物种头尾适配：

从仓库根目录执行：

```bash
python SimCLR/run.py --data-root data/tomato --output-dir outputs/clr_source
python SimCLR/train.py --species corn --data-root data/corn --checkpoint outputs/clr_source/source.pth
```

预训练默认 1000 轮、温度 0.05、batch size 64；投影 MLP 为 512→512→128。下游支持三种物种、五个种子、头尾消融与标注比例实验，保留原始目录式随机划分方案。

数据、权重、结果均留在本地。完整命令和论文一致性边界见 [主说明](../README.md)。
