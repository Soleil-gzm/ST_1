#!/usr/bin/env python3
"""
降维 + 聚类
输入：outputs/{timestamp}/embeddings.npy
输出：outputs/{timestamp}/reduced_embeddings.npy, outputs/{timestamp}/cluster_labels.npy
"""

import numpy as np
import umap
import hdbscan
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
    emb_path = input_dir / "embeddings.npy"
    if not emb_path.exists():
        print(f"错误：{emb_path} 不存在，请先运行向量化脚本")
        return False

    embeddings = np.load(emb_path)
    print(f"加载向量，形状 {embeddings.shape}")

    print("降维中...")
    reducer = umap.UMAP(n_components=15, random_state=42, n_neighbors=30, min_dist=0.0)
    reduced = reducer.fit_transform(embeddings)
    reduced_path = input_dir / "reduced_embeddings.npy"
    np.save(reduced_path, reduced)
    print(f"降维完成，形状 {reduced.shape}，保存至 {reduced_path}")

    print("聚类中...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, prediction_data=True)
    labels = clusterer.fit_predict(reduced)
    labels_path = input_dir / "cluster_labels.npy"
    np.save(labels_path, labels)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    print(f"聚类完成，簇数量（不含噪声）: {n_clusters}")
    print(f"噪声点数量: {n_noise} ({n_noise/len(labels)*100:.2f}%)")
    print(f"标签保存至 {labels_path}")
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