#!/usr/bin/env python3
"""
评估 K-means 聚类的最佳 K 值（肘部法则）
输入：outputs/{timestamp}_text2vec_kmeans/embeddings_pca.npy
输出：outputs/{timestamp}_text2vec_kmeans/kmeans_elbow.png
      outputs/{timestamp}_text2vec_kmeans/kmeans_inertia.csv
"""

'''
解读肘部图
肘点：曲线从急剧下降变为平缓下降的转折点。该点对应的 K 值通常是最佳选择。
如果曲线没有明显肘点：可能数据自然簇数较少或分布均匀，可结合业务目标选择 K（例如期望 500 个簇）。
示例：若曲线在 K=500 后下降缓慢，则 K=500 是较优选择。
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pathlib import Path
import sys
import time

# ========== 配置 ==========
INPUT_TIMESTAMP = None          # None=自动使用最新含 _text2vec_kmeans 的目录
K_VALUES = [100, 200, 500, 1000, 1500, 2000, 2500]  # 测试的 K 值列表
# ========================

def get_latest_kmeans_output(output_dir="outputs"):
    """获取最新的 _text2vec_kmeans 输出目录"""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    dirs = [d for d in output_path.iterdir() if d.is_dir() and '_text2vec_kmeans' in d.name]
    if not dirs:
        return None
    dirs.sort(reverse=True)
    return dirs[0]

def main():
    # 确定输入目录
    if INPUT_TIMESTAMP is None:
        input_dir = get_latest_kmeans_output()
        if input_dir is None:
            print("错误：未找到任何 _text2vec_kmeans 输出目录，请先运行聚类脚本")
            sys.exit(1)
        print(f"自动使用最新输出目录: {input_dir}")
    else:
        input_dir = Path("outputs") / INPUT_TIMESTAMP
        if not input_dir.exists():
            print(f"错误：目录 {input_dir} 不存在")
            sys.exit(1)

    # 加载降维后的向量
    emb_path = input_dir / "embeddings_pca.npy"
    if not emb_path.exists():
        print(f"错误：{emb_path} 不存在，请先运行聚类脚本")
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
    csv_path = input_dir / "kmeans_inertia.csv"
    df.to_csv(csv_path, index=False)
    print(f"已保存 inertia 数据到 {csv_path}")

    # 绘制肘部图
    plt.figure(figsize=(10, 6))
    plt.plot(k_list, inertia_list, 'bo-', linewidth=2, markersize=8)
    plt.xlabel("Number of clusters (K)", fontsize=12)
    plt.ylabel("Inertia (Sum of squared distances)", fontsize=12)
    plt.title("Elbow Method for Optimal K", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    # 标注可能的肘点（目测）
    # 计算每个点的二阶导数近似，找曲率最大点（可选）
    plt.tight_layout()
    png_path = input_dir / "kmeans_elbow.png"
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