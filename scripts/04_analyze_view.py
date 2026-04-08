#!/usr/bin/env python3
"""
分析聚类结果，生成 CSV 报告和可视化散点图（静态 PNG + 交互式 HTML）
输入：outputs/{timestamp}/cleaned_intents.txt, outputs/{timestamp}/embeddings.npy,
      outputs/{timestamp}/reduced_embeddings.npy, outputs/{timestamp}/cluster_labels.npy
输出：outputs/{timestamp}/clustered_intents.csv, outputs/{timestamp}/cluster_summary.csv,
      outputs/{timestamp}/cluster_visualization.png,
      outputs/{timestamp}/cluster_visualization_interactive.html
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from pathlib import Path
import sys

# ========== 配置 ==========
TARGET_TIMESTAMP = None
# ========================

def get_latest_timestamp(output_dir="outputs"):
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    timestamps = [d.name for d in output_path.iterdir() if d.is_dir()]
    if not timestamps:
        return None
    timestamps.sort(reverse=True)
    return timestamps[0]

def main(timestamp):
    input_dir = Path("outputs") / timestamp
    texts_path = input_dir / "cleaned_intents.txt"
    labels_path = input_dir / "cluster_labels.npy"
    reduced_path = input_dir / "reduced_embeddings.npy"
    emb_path = input_dir / "embeddings.npy"

    for p in [texts_path, labels_path, reduced_path]:
        if not p.exists():
            print(f"错误：{p} 不存在，请先运行前面的脚本")
            return False

    with open(texts_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    labels = np.load(labels_path)
    reduced = np.load(reduced_path)
    embeddings = np.load(emb_path)

    # 1. 保存带标签的 CSV
    df = pd.DataFrame({"intent": texts, "cluster_id": labels})
    csv_path = input_dir / "clustered_intents.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"已保存 {csv_path}")

    # 2. 生成簇汇总
    cluster_summary = []
    unique_clusters = sorted(set(labels))
    for cid in unique_clusters:
        if cid == -1:
            continue
        indices = np.where(labels == cid)[0]
        size = len(indices)
        cluster_center = np.mean(embeddings[indices], axis=0)
        dists = np.linalg.norm(embeddings[indices] - cluster_center, axis=1)
        best_idx = indices[np.argmin(dists)]
        representative = texts[best_idx]
        cluster_summary.append({
            "cluster_id": cid,
            "size": size,
            "representative_text": representative
        })
    cluster_summary.sort(key=lambda x: x["size"], reverse=True)
    summary_df = pd.DataFrame(cluster_summary)
    summary_path = input_dir / "cluster_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"已保存 {summary_path}")

    # 3. 可视化：降维到 2D（用于绘图）
    print("生成可视化图片...")
    reducer_2d = umap.UMAP(n_components=2, random_state=42)
    reduced_2d = reducer_2d.fit_transform(embeddings)

    # ---------- 静态 PNG（使用丰富颜色）----------
    plt.figure(figsize=(16, 12))
    # 获取所有簇标签（包括 -1）
    unique_labels = np.unique(labels)
    # 为每个簇分配一个随机颜色（但保持噪声为灰色）
    import random
    random.seed(42)
    colors = {}
    for label in unique_labels:
        if label == -1:
            colors[label] = 'gray'
        else:
            # 生成随机 RGB 颜色，保证亮度较高（避免太暗）
            colors[label] = (random.random(), random.random(), random.random())
    
    for label in unique_labels:
        mask = labels == label
        color = colors[label]
        plt.scatter(reduced_2d[mask, 0], reduced_2d[mask, 1],
                    c=[color], s=1, alpha=0.6, label=f'Cluster {label}' if label != -1 else 'Noise')
    plt.title("Intent Clustering Visualization (UMAP + HDBSCAN)")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    # 图例太多会重叠，只显示前20个簇
    handles, labels_leg = plt.gca().get_legend_handles_labels()
    if len(handles) > 25:
        plt.legend(handles[:25], labels_leg[:25], loc='upper right', fontsize=8)
    else:
        plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    png_path = input_dir / "cluster_visualization.png"
    plt.savefig(png_path, dpi=200)
    plt.close()
    print(f"已保存静态 PNG: {png_path}")

    # ---------- 交互式 HTML (plotly) ----------
    try:
        import plotly.express as px
        # 构建 DataFrame
        df_plot = pd.DataFrame({
            'x': reduced_2d[:, 0],
            'y': reduced_2d[:, 1],
            'cluster': labels.astype(str),
            'intent': texts
        })
        # 将噪声点的 cluster 改为 "Noise"
        df_plot.loc[df_plot['cluster'] == '-1', 'cluster'] = 'Noise'
        # 使用 plotly 绘制，颜色使用离散色板（自动区分）
        fig = px.scatter(df_plot, x='x', y='y', color='cluster',
                         hover_data=['intent'],
                         title=f'Intent Clustering (timestamp: {timestamp})',
                         labels={'cluster': 'Cluster ID'},
                         color_discrete_sequence=px.colors.qualitative.Prism)  # 使用丰富色板
        fig.update_traces(marker=dict(size=2, opacity=0.6))
        html_path = input_dir / "cluster_visualization_interactive.html"
        fig.write_html(html_path)
        print(f"已保存交互式 HTML: {html_path}")
    except ImportError:
        print("警告: 未安装 plotly，跳过生成交互式 HTML。可运行 pip install plotly 安装。")

    # 统计信息
    print("\n=== 聚类统计 ===")
    print(f"总样本数: {len(texts)}")
    print(f"簇数量（不含噪声）: {len(unique_clusters) - (1 if -1 in unique_clusters else 0)}")
    print(f"噪声样本数: {np.sum(labels == -1)} ({np.sum(labels == -1)/len(labels)*100:.2f}%)")
    if not summary_df.empty:
        print(f"最大簇大小: {summary_df['size'].max()}")
        print(f"最小簇大小: {summary_df['size'].min()}")
    return True

if __name__ == "__main__":
    if TARGET_TIMESTAMP is None:
        timestamp = get_latest_timestamp()
        if timestamp is None:
            print("错误：未找到任何时间戳目录，请先运行清洗脚本")
            sys.exit(1)
        print(f"自动使用最新时间戳: {timestamp}")
    else:
        timestamp = TARGET_TIMESTAMP
        print(f"使用固定时间戳: {timestamp}")
    main(timestamp)


