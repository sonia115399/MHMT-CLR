# MHMT-Net

本目录对应 MHMT-Net，同时提供作者原始下游代码与按论文附录整理的统一入口。

完整头尾训练代码为 `light_corn.py`、`light_datepalmhead.py`、`light_tea.py`，已按手稿补齐下游 stem、固定编码器参数与 BN，并调整椰枣 SE 位置；原始尾部训练代码为 `corn_cnn.py`、`datepalm_cnn.py`、`tea_cnn.py`。依赖位于 `model/`、`dataprocess/`、`utils/`。完整网络类和训练流程保留在文件内，具体运行方式见 [原始下游脚本](../README.md#原始下游脚本)。

下列命令为统一入口，共用实现位于 `../mhmt/`，编码器采用番茄源域监督预训练：

从仓库根目录执行：

```bash
python code/run.py --data-root data/tomato --output-dir outputs/net_source
python code/train.py --species corn --data-root data/corn --checkpoint outputs/net_source/source.pth
```

`--species` 支持 `corn`、`datepalm`、`tea`；默认运行五个种子。下游保留已有 train/test 目录和原始 80%/20% 随机训练/验证划分。数据、权重、结果均留在本地。

完整安装、数据结构、物种设置、评估方式和论文一致性边界见 [主说明](../README.md)。
