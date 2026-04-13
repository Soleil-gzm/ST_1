#!/usr/bin/env python3
"""
评估 K-means 聚类的最佳 K 值（肘部法则）
输入：outputs/{timestamp}_base_embeddings/embeddings_pca.npy
输出：outputs/{timestamp}_base_embeddings/kmeans_inertia.csv
      outputs/{timestamp}_base_embeddings/kmeans_elbow.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pathlib import Path
import sys
import time

# ========== 配置 ==========
# 可手动指定 base 目录，None 表示自动使用最新的 _base_embeddings 目录
BASE_DIR = None
# 测试的 K 值列表（可根据需要调整）
K_VALUES = [100, 200, 300,400,500,600,700,800, 1000]
# K_VALUES = [100, 200]
# ========================

def get_latest_base_dir(output_dir="outputs"):
    """获取最新的以 _base_embeddings 结尾的目录"""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    dirs = [d for d in output_path.iterdir() if d.is_dir() and d.name.endswith('_base_embeddings')]
    if not dirs:
        return None
    dirs.sort(reverse=True)
    return dirs[0]

def main():
    # 确定输入目录
    if BASE_DIR is None:
        base_dir = get_latest_base_dir()
        if base_dir is None:
            print("错误：未找到任何 _base_embeddings 目录，请先运行 02_generate_embeddings.py")
            sys.exit(1)
        print(f"自动使用最新 base 目录: {base_dir}")
    else:
        base_dir = Path(BASE_DIR)
        if not base_dir.exists():
            print(f"错误：目录 {base_dir} 不存在")
            sys.exit(1)

    # 加载降维后的向量
    emb_path = base_dir / "embeddings_pca.npy"
    if not emb_path.exists():
        print(f"错误：{emb_path} 不存在，请先运行 02_generate_embeddings.py")
        sys.exit(1)
    X = np.load(emb_path)
    print(f"加载数据形状: {X.shape}")

    # 存储结果
    inertia_list = []
    k_list = []

    print("开始评估不同 K 值...")
    for k in K_VALUES:
        print(f"  运行 K = {k} ...", end="", flush=True)
        start = time.time()
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=300, verbose=0)
        kmeans.fit(X)
        inertia = kmeans.inertia_
        inertia_list.append(inertia)
        k_list.append(k)
        elapsed = time.time() - start
        print(f" 完成，耗时 {elapsed:.1f}s，inertia = {inertia:.2f}")

    # 保存 CSV
    df = pd.DataFrame({"K": k_list, "inertia": inertia_list})
    csv_path = base_dir / "kmeans_inertia.csv"
    df.to_csv(csv_path, index=False)
    print(f"已保存 inertia 数据到 {csv_path}")

    # 绘制肘部图
    plt.figure(figsize=(10, 6))
    plt.plot(k_list, inertia_list, 'bo-', linewidth=2, markersize=8)
    plt.xlabel("Number of clusters (K)", fontsize=12)
    plt.ylabel("Inertia (Sum of squared distances)", fontsize=12)
    plt.title("Elbow Method for Optimal K", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    png_path = base_dir / "kmeans_elbow.png"
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"已保存肘部图到 {png_path}")

    # 输出建议
    print("\n=== 建议 ===")
    print("观察肘部图，选择曲线拐点处的 K 值。")
    print("如果曲线平滑下降，可考虑 K = {} 附近。".format(k_list[inertia_list.index(min(inertia_list))]))
    print("你也可以结合业务需求（例如期望的簇数量）进一步调整。")

if __name__ == "__main__":
    main()