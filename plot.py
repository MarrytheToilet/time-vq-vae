import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def extract_features(loader, model, device):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for data, label in loader:
            data = data.to(device)
            max_len = 134
            data = data[:, :, :max_len]
            _, quant_b, _, _, _ = model.encode(data)
            features.append(quant_b.cpu().numpy())
            labels.append(label.cpu().numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

def plot_embedding(features, labels, title, filename):
    plt.figure(figsize=(10, 10))
    
    # 使用seaborn的scatterplot绘制散点图
    palette = sns.color_palette("tab10", n_colors=len(set(labels)))
    scatter = sns.scatterplot(x=features[:, 0], y=features[:, 1], hue=labels, palette=palette, s=40, edgecolor='w', alpha=0.7)
    
    # 设置标题和轴标签
    plt.title(title, fontsize=20, weight='bold')
    plt.xlabel("Component 1", fontsize=15)
    plt.ylabel("Component 2", fontsize=15)
    
    # 美化图例
    legend = scatter.legend(title='Class', title_fontsize='13', fontsize='11', loc='best', borderpad=1, frameon=True)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(0.8)
    
    # 美化网格
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 美化轴
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.2)
    plt.gca().spines['bottom'].set_linewidth(1.2)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()