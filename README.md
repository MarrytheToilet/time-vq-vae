# time-VQ-VAE-2 ECG5000

本项目实现了 time-VQ-VAE-2 模型，并应用于 ECG5000 数据集。项目包含数据集加载、模型训练、特征提取和可视化等功能。

## 项目结构

- `dataset.py`：包含数据集类 `ECGDataset` 的定义。
- `model.py`：包含模型架构，基本结构参考自 [rosinality/vq-vae-2-pytorch](https://github.com/rosinality/vq-vae-2-pytorch)。
- `train.ipynb`：用于模型训练的 Jupyter Notebook，可以根据需要调整参数。
- `plot.py`：包含特征提取和绘图的方法。
- `plot.ipynb`：用于测试特征提取和绘图的 Jupyter Notebook。
- `data`：包含原始数据文件 `ECG5000_TRAIN.txt` 和 `ECG5000_TEST.txt`。
- `pca_sample`、`tsne_sample`：每100个epoch保存一次的特征可视化绘图结果。
- `tsne_progress.gif`、`pca_progress.gif`：特征变化的 GIF 动图。

## 数据集

使用的原始数据集为 ECG5000，包括：
- 500 个训练样本 (`ECG5000_TRAIN.txt`)
- 4500 个测试样本 (`ECG5000_TEST.txt`)

## 运行环境

请确保已经安装以下依赖项：

```bash
pip install torch torchvision tqdm matplotlib seaborn scikit-learn umap-learn imageio
