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
# 输入时间戳：从哪里读取 embeddings.npy
INPUT_TIMESTAMP = None          # None=自动使用最新，或指定如 "20260407_174104"
# 输出时间戳：结果保存到哪个目录（建议不同参数使用不同后缀）
OUTPUT_TIMESTAMP = None         # None=自动生成（输入时间戳 + "_cluster"），或手动指定如 "20260407_174104_min25"
# ========================

#获取 outputs/ 目录下最新的时间戳子目录
def get_latest_timestamp(output_dir="outputs"):
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    timestamps = [d.name for d in output_path.iterdir() if d.is_dir()]
    if not timestamps:
        return None
    timestamps.sort(reverse=True)
    return timestamps[0]

# def main(timestamp):
#     input_dir = Path("outputs") / timestamp
#     emb_path = input_dir / "embeddings.npy"
#     if not emb_path.exists():
#         print(f"错误：{emb_path} 不存在，请先运行向量化脚本")
#         return False

#     embeddings = np.load(emb_path)
#     print(f"加载向量，形状 {embeddings.shape}")

def main(input_ts, output_ts):
    input_dir = Path("outputs") / input_ts
    emb_path = input_dir / "embeddings.npy"
    if not emb_path.exists():
        print(f"错误：{emb_path} 不存在，请先运行向量化脚本")
        return False

    output_dir = Path("outputs") / output_ts
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")

    embeddings = np.load(emb_path)
    print(f"加载向量，形状 {embeddings.shape}")

    ''' 
    UMAP 降维：
    n_components=15：降维后的目标维度，从 384 维降到 15 维。15 是经验值，既能保留大部分结构，又方便聚类。
    random_state=42：固定随机种子，保证结果可重复。
    n_neighbors=30：UMAP 构建近邻图时考虑的邻居数量，值越大越注重全局结构，值越小越注重局部结构。通常设为 15~30。
    min_dist=0.0：降维后点之间的最小距离，0.0 表示允许非常接近，适合聚类。
    fit_transform(embeddings)：同时训练并转换，得到降维后的数组 reduced，形状 (样本数, 15)。
    '''

    '''
    调整 UMAP 参数（影响形状）
    在 03_cluster.py 中修改 reducer：

    参数	            当前值	            建议尝试	                        效果
    n_neighbors	        30	               15 或 50	                越小越注重局部结构（点更聚集，簇更紧凑）；越大越注重全局结构（长条状更明显）。
    min_dist	        0.0	               0.1 或 0.5	            越大点越分散，簇之间空隙更大，减少重叠。
    n_components	    15	               10 或 25	                降低维度（10）可能丢失信息，但计算更快；增加维度（25）保留更多细节，但聚类可能更复杂。

    推荐组合（减少重叠和长条状）：reducer = umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.1, random_state=42)
    '''

    print("降维中...")
    reducer = umap.UMAP(n_components=15, random_state=42, n_neighbors=30, min_dist=0.0)
    reduced = reducer.fit_transform(embeddings)
    reduced_path = output_dir / "reduced_embeddings.npy"
    np.save(reduced_path, reduced)
    print(f"降维完成，形状 {reduced.shape}，保存至 {reduced_path}")

    '''
    HDBSCAN 聚类：
    min_cluster_size=15：簇的最小规模。小于此值的密集区域被视为噪声（标签 -1）。增大此值会减少簇的数量，增大噪声比例。
    prediction_data=True：保存预测数据，以便将来将新样本分配到已有簇（本脚本未使用，但保留扩展性）。
    fit_predict(reduced)：对降维后的数据进行聚类，返回每个样本的簇标签（整数，-1 表示噪声点）。
    '''
    '''
    调整HDBSCAN 参数（影响簇的数量和噪声）
    参数	                当前值	            建议尝试	                效果
    min_cluster_size	    15	              10 或 25	        减小会得到更多小簇（孤立的彩色点增多）；增大则合并小簇，减少颜色种类。
    min_samples	            未设置	           5 或 10	         控制噪声点的灵敏度。值越大，点更容易被标记为噪声。

    推荐组合（减少孤立小簇）：clusterer = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=5, prediction_data=True)
    '''

    print("聚类中...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=25, prediction_data=True)  # 调整这里
    labels = clusterer.fit_predict(reduced)
    labels_path = output_dir / "cluster_labels.npy"
    np.save(labels_path, labels)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    print(f"聚类完成，簇数量（不含噪声）: {n_clusters}")
    print(f"噪声点数量: {n_noise} ({n_noise/len(labels)*100:.2f}%)")
    print(f"标签保存至 {labels_path}")
    return True

if __name__ == "__main__":
    # 确定输入时间戳
    if INPUT_TIMESTAMP is None:
        input_ts = get_latest_timestamp()
        if input_ts is None:
            print("错误：未找到任何时间戳目录，请先运行清洗脚本")
            exit(1)
        print(f"自动使用最新输入时间戳: {input_ts}")
    else:
        input_ts = INPUT_TIMESTAMP
        print(f"使用指定输入时间戳: {input_ts}")

    # 确定输出时间戳
    if OUTPUT_TIMESTAMP is None:
        output_ts = input_ts + "_cluster"   # 在原时间戳后加后缀
        print(f"自动生成输出时间戳: {output_ts}")
    else:
        output_ts = OUTPUT_TIMESTAMP
        print(f"使用指定输出时间戳: {output_ts}")

    main(input_ts, output_ts)


    '''
    该脚本将高维向量（384维）通过 UMAP 降维到 15 维，然后使用 HDBSCAN 聚类，自动确定簇的数量并标记噪声点。
    最终输出降维后的向量和聚类标签，供后续分析可视化使用。
    运行时间主要消耗在 UMAP 的 k-近邻图构建上，数据量较大时可能较慢，但属于正常现象。
    '''