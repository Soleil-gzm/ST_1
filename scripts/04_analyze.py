#!/usr/bin/env python3
"""
分析聚类结果，生成 CSV 报告和可视化散点图
输入：outputs/{timestamp}/cleaned_intents.txt, outputs/{timestamp}/embeddings.npy,
      outputs/{timestamp}/reduced_embeddings.npy, outputs/{timestamp}/cluster_labels.npy
输出：outputs/{timestamp}/clustered_intents.csv, outputs/{timestamp}/cluster_summary.csv,
      outputs/{timestamp}/cluster_visualization.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from pathlib import Path

# ========== 配置 ==========
# 设置为 None 表示自动使用最新的时间戳目录
# 如果你想固定使用某个时间戳，直接写字符串，例如 "20260407_161619"
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

    # 3. 可视化（降维到 2D）
    print("生成可视化图片...")
    reducer_2d = umap.UMAP(n_components=2, random_state=42)
    reduced_2d = reducer_2d.fit_transform(embeddings)

    plt.figure(figsize=(16, 12))
    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap('tab20', len(unique_labels))
    for label in unique_labels:
        mask = labels == label
        color = 'gray' if label == -1 else cmap(label % 20)
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
    png_path = input_dir / "cluster_visualization.png"
    plt.savefig(png_path, dpi=200)
    plt.close()
    print(f"已保存 {png_path}")

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
            exit(1)
        print(f"自动使用最新时间戳: {timestamp}")
    else:
        timestamp = TARGET_TIMESTAMP
        print(f"使用固定时间戳: {timestamp}")
    main(timestamp)