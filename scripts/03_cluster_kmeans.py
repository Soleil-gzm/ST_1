#!/usr/bin/env python3
"""
读取已保存的降维向量，执行 K-means 聚类（可多次运行调整参数）
输入：outputs/{input_base}/embeddings_pca.npy
输出：outputs/{input_base}_kmeans_K{cluster_num}/cluster_labels.npy
      outputs/{input_base}_kmeans_K{cluster_num}/clustered_intents.csv
      outputs/{input_base}_kmeans_K{cluster_num}/cluster_summary.csv (可选)
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from pathlib import Path
import sys

# ========== 配置 ==========
INPUT_BASE = None               # None=自动使用最新的 _base_embeddings 目录
N_CLUSTERS = 1000               # K-means 簇数量，可修改
# ========================

def get_latest_base_dir(output_dir="outputs"):
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    # dirs = [d for d in output_path.iterdir() if d.is_dir() and '_base_embeddings' in d.name]
    dirs = [d for d in output_path.iterdir() if d.is_dir() and d.name.endswith('_base_embeddings')]
    if not dirs:
        return None
    dirs.sort(reverse=True)
    return dirs[0]

def main():
    # 确定输入目录
    if INPUT_BASE is None:
        base_dir = get_latest_base_dir()
        if base_dir is None:
            print("错误：未找到任何 _base_embeddings 目录，请先运行 02_generate_embeddings.py")
            sys.exit(1)
        print(f"自动使用最新 base 目录: {base_dir.name}")
    else:
        base_dir = Path("outputs") / INPUT_BASE
        if not base_dir.exists():
            print(f"错误：目录 {base_dir} 不存在")
            sys.exit(1)

    # 加载降维后的向量
    pca_path = base_dir / "embeddings_pca.npy"
    if not pca_path.exists():
        print(f"错误：{pca_path} 不存在")
        return
    X = np.load(pca_path)
    print(f"加载向量形状: {X.shape}")

    # K-means 聚类
    print(f"执行 K-means 聚类，簇数 = {N_CLUSTERS} ...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10, verbose=0)
    labels = kmeans.fit_predict(X)
    print("聚类完成")

    # 创建输出目录（基于 base_dir 名称，加上 K 值后缀）
    output_dir_name = base_dir.name + f"_kmeans_K{N_CLUSTERS}"
    output_dir = base_dir.parent / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存标签
    np.save(output_dir / "cluster_labels.npy", labels)

    # 读取原始文本（从 base_dir 对应的原始数据目录获取）
    # base_dir 名称形如 "20260409_131912_base_embeddings"，原始目录为 "20260409_131912"
    raw_timestamp = base_dir.name.replace("_base_embeddings", "")
    raw_dir = base_dir.parent / raw_timestamp
    txt_path = raw_dir / "cleaned_intents.txt"
    if not txt_path.exists():
        print(f"错误：找不到原始文本文件 {txt_path}")
        return
    with open(txt_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]

    # 保存带标签的 CSV
    df = pd.DataFrame({"intent": texts, "cluster_id": labels})
    csv_path = output_dir / "clustered_intents.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"已保存 {csv_path}")

    # 简单统计
    unique, counts = np.unique(labels, return_counts=True)
    print("\n=== 聚类统计 ===")
    print(f"总样本数: {len(texts)}")
    print(f"实际簇数量: {len(unique)}")
    print(f"最大簇大小: {counts.max()}")
    print(f"最小簇大小: {counts.min()}")
    print(f"平均簇大小: {counts.mean():.1f}")
    print(f"输出目录: {output_dir}")

if __name__ == "__main__":
    main()