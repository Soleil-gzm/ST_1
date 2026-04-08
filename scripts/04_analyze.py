#!/usr/bin/env python3
"""
分析聚类结果，生成 CSV 报告和可视化
输入：outputs/{input_timestamp}/cleaned_intents.txt, outputs/{input_timestamp}/embeddings.npy,
      outputs/{output_timestamp}/reduced_embeddings.npy, outputs/{output_timestamp}/cluster_labels.npy
输出：outputs/{output_timestamp}/clustered_intents.csv, outputs/{output_timestamp}/cluster_summary.csv,
      outputs/{output_timestamp}/cluster_visualization.png,
      outputs/{output_timestamp}/cluster_visualization_interactive.html
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from pathlib import Path
import sys
import random

# ========== 配置 ==========
# 输入时间戳：原始数据目录（包含 cleaned_intents.txt 和 embeddings.npy）
INPUT_TIMESTAMP = None          # None=自动使用最新（排除 _cluster 后缀）
# 输出时间戳：聚类结果目录（包含 reduced_embeddings.npy 和 cluster_labels.npy）
OUTPUT_TIMESTAMP = None         # None=自动生成（输入时间戳 + "_cluster"）
# ========================

def get_latest_input_timestamp(output_dir="outputs"):
    """获取最新的原始数据目录（排除包含 '_cluster' 的目录）"""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    timestamps = [d.name for d in output_path.iterdir() if d.is_dir() and '_cluster' not in d.name]
    if not timestamps:
        return None
    timestamps.sort(reverse=True)
    return timestamps[0]

def main(input_ts, output_ts):
    input_dir = Path("outputs") / input_ts
    output_dir = Path("outputs") / output_ts

    texts_path = input_dir / "cleaned_intents.txt"
    labels_path = output_dir / "cluster_labels.npy"
    reduced_path = output_dir / "reduced_embeddings.npy"
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
    csv_path = output_dir / "clustered_intents.csv"
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
    summary_path = output_dir / "cluster_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"已保存 {summary_path}")

    # 3. 可视化：降维到 2D（用于绘图）
    print("生成可视化图片...")
    reducer_2d = umap.UMAP(n_components=2, random_state=42)
    reduced_2d = reducer_2d.fit_transform(embeddings)

    # 静态 PNG
    plt.figure(figsize=(16, 12))
    unique_labels = np.unique(labels)
    random.seed(42)
    colors = {}
    for label in unique_labels:
        if label == -1:
            colors[label] = 'gray'
        else:
            colors[label] = (random.random(), random.random(), random.random())
    for label in unique_labels:
        mask = labels == label
        color = colors[label]
        plt.scatter(reduced_2d[mask, 0], reduced_2d[mask, 1],
                    c=[color], s=1, alpha=0.6, label=f'Cluster {label}' if label != -1 else 'Noise')
    plt.title("Intent Clustering Visualization (UMAP + HDBSCAN)")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    handles, labels_leg = plt.gca().get_legend_handles_labels()
    if len(handles) > 25:
        plt.legend(handles[:25], labels_leg[:25], loc='upper right', fontsize=8)
    else:
        plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    png_path = output_dir / "cluster_visualization.png"
    plt.savefig(png_path, dpi=200)
    plt.close()
    print(f"已保存静态 PNG: {png_path}")

    # 交互式 HTML
    try:
        import plotly.express as px
        df_plot = pd.DataFrame({
            'x': reduced_2d[:, 0],
            'y': reduced_2d[:, 1],
            'cluster': labels.astype(str),
            'intent': texts
        })
        df_plot.loc[df_plot['cluster'] == '-1', 'cluster'] = 'Noise'
        fig = px.scatter(df_plot, x='x', y='y', color='cluster',
                         hover_data=['intent'],
                         title=f'Intent Clustering (input: {input_ts}, output: {output_ts})',
                         labels={'cluster': 'Cluster ID'},
                         color_discrete_sequence=px.colors.qualitative.Prism)
        fig.update_traces(marker=dict(size=2, opacity=0.6))
        html_path = output_dir / "cluster_visualization_interactive.html"
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
    if INPUT_TIMESTAMP is None:
        input_ts = get_latest_input_timestamp()
        if input_ts is None:
            print("错误：未找到任何输入时间戳目录（不含 _cluster 后缀），请先运行清洗和向量化脚本")
            sys.exit(1)
        print(f"自动使用最新输入时间戳: {input_ts}")
    else:
        input_ts = INPUT_TIMESTAMP
        print(f"使用指定输入时间戳: {input_ts}")

    if OUTPUT_TIMESTAMP is None:
        output_ts = input_ts + "_cluster"
        print(f"自动生成输出时间戳: {output_ts}")
    else:
        output_ts = OUTPUT_TIMESTAMP
        print(f"使用指定输出时间戳: {output_ts}")

    main(input_ts, output_ts)