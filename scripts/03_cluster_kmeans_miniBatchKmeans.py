#!/usr/bin/env python3
"""
读取已保存的降维向量，执行 K-means 聚类（可多次运行调整参数）
支持手动指定 K 值或自动使用肘部法则选择最优 K。
输入：outputs/{input_base}/embeddings_pca.npy
输出：outputs/{input_base}_kmeans_K{cluster_num}/cluster_labels.npy
      outputs/{input_base}_kmeans_K{cluster_num}/clustered_intents.csv
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from pathlib import Path
import sys
import argparse
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# 尝试导入 kneed 库，用于自动选 K
try:
    from kneed import KneeLocator
    KNEE_AVAILABLE = True
except ImportError:
    KNEE_AVAILABLE = False

# ========== 默认配置（可通过命令行覆盖） ==========
DEFAULT_N_CLUSTERS = 250       # 手动模式下的默认簇数
DEFAULT_K_RANGE_START = 450      # 自动选 K 的搜索起始值
DEFAULT_K_RANGE_END = 800      # 自动选 K 的搜索结束值
DEFAULT_RANDOM_STATE = 42
DEFAULT_BATCH_SIZE = 10000     # MiniBatchKMeans 的批大小
# ================================================

# 全局变量，用于并行任务共享数据（避免序列化大数组）
_X_GLOBAL = None
_RANDOM_STATE_GLOBAL = None
_N_INIT_GLOBAL = None
_MAX_ITER_GLOBAL = None
_BATCH_SIZE_GLOBAL = None
_USE_MINI_BATCH_GLOBAL = None

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

def _evaluate_single_k(k):
    """在单个进程中评估一个 K 值，返回 (k, inertia)"""
    if _USE_MINI_BATCH_GLOBAL:
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            random_state=_RANDOM_STATE_GLOBAL,
            n_init=_N_INIT_GLOBAL,
            max_iter=_MAX_ITER_GLOBAL,
            batch_size=_BATCH_SIZE_GLOBAL,
            verbose=0
        )
    else:
        kmeans = KMeans(
            n_clusters=k,
            random_state=_RANDOM_STATE_GLOBAL,
            n_init=_N_INIT_GLOBAL,
            max_iter=_MAX_ITER_GLOBAL,
            verbose=0
        )
    kmeans.fit(_X_GLOBAL)
    return k, kmeans.inertia_

def find_optimal_k_by_elbow(X, k_range_start, k_range_end, random_state,
                            n_init=10, max_iter=300, use_minibatch=True,
                            batch_size=10000, parallel=True, n_jobs=None):
    """
    使用肘部法则自动寻找最佳 K 值（支持 MiniBatch 和并行计算）。
    
    Args:
        X: 特征矩阵
        k_range_start, k_range_end: K 值范围
        random_state: 随机种子
        n_init, max_iter: KMeans 参数
        use_minibatch: 是否使用 MiniBatchKMeans（加速）
        batch_size: MiniBatchKMeans 的批大小
        parallel: 是否并行计算（多个 K 同时跑）
        n_jobs: 并行进程数，默认为 CPU 核心数
    
    Returns:
        optimal_k, inertias (列表，顺序与 k_range_start..k_range_end 一致)
    """
    global _X_GLOBAL, _RANDOM_STATE_GLOBAL, _N_INIT_GLOBAL, _MAX_ITER_GLOBAL
    global _BATCH_SIZE_GLOBAL, _USE_MINI_BATCH_GLOBAL
    
    # 设置全局变量供子进程使用
    _X_GLOBAL = X
    _RANDOM_STATE_GLOBAL = random_state
    _N_INIT_GLOBAL = n_init
    _MAX_ITER_GLOBAL = max_iter
    _BATCH_SIZE_GLOBAL = batch_size
    _USE_MINI_BATCH_GLOBAL = use_minibatch
    
    k_values = list(range(k_range_start, k_range_end + 1))
    inertias = [None] * len(k_values)  # 预分配，保持顺序
    
    if not KNEE_AVAILABLE:
        raise ImportError("kneed 库未安装，无法进行自动选 K。请运行 'pip install kneed' 安装。")
    
    print(f"开始自动评估 K 值，范围 [{k_range_start}, {k_range_end}] ...")
    start_time = time.time()
    
    if parallel:
        # 并行执行
        n_jobs = n_jobs or multiprocessing.cpu_count()
        print(f"使用并行模式，进程数: {n_jobs}，{'MiniBatch' if use_minibatch else '标准'}KMeans")
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # 提交所有任务
            future_to_k = {executor.submit(_evaluate_single_k, k): k for k in k_values}
            # 收集结果
            for future in as_completed(future_to_k):
                k, inertia = future.result()
                idx = k - k_range_start
                inertias[idx] = inertia
                print(f"  K = {k} 完成，inertia = {inertia:.2f}")
    else:
        # 串行模式
        print(f"使用串行模式，{'MiniBatch' if use_minibatch else '标准'}KMeans")
        for i, k in enumerate(k_values):
            print(f"  正在评估 K = {k}...", end="", flush=True)
            _, inertia = _evaluate_single_k(k)
            inertias[i] = inertia
            print(f" 完成，inertia = {inertia:.2f}")
    
    elapsed = time.time() - start_time
    print(f"自动评估总耗时: {elapsed:.1f} 秒")
    
    # 使用 kneed 检测肘部点
    kn = KneeLocator(
        x=k_values,
        y=inertias,
        curve='convex',
        direction='decreasing',
        online=False
    )
    optimal_k = kn.knee
    if optimal_k is None:
        # 回退策略：选择 inertia 下降率变化最大的点（二阶差分最大）
        print("警告: kneed 未检测到明显肘点，将使用 inertia 下降率变化最大的点作为备选。")
        diffs = np.diff(inertias)
        if len(diffs) > 1:
            second_diffs = np.diff(diffs)
            idx = np.argmax(second_diffs)
            optimal_k = k_range_start + idx + 1
        else:
            optimal_k = k_range_start
        print(f"回退策略选择的 K 值为: {optimal_k}")
    else:
        print(f"kneed 库检测到最佳 K 值为: {optimal_k}")
    
    return optimal_k, inertias

def main():
    parser = argparse.ArgumentParser(description="K-means 意图聚类脚本，支持手动指定 K 值或自动肘部法则选 K")
    parser.add_argument('--auto_k', action='store_true',
                        help='启用自动选择最优 K 值（使用肘部法则）')
    parser.add_argument('--k_range_start', type=int, default=DEFAULT_K_RANGE_START,
                        help=f'自动选 K 的搜索起始值（默认 {DEFAULT_K_RANGE_START}）')
    parser.add_argument('--k_range_end', type=int, default=DEFAULT_K_RANGE_END,
                        help=f'自动选 K 的搜索结束值（默认 {DEFAULT_K_RANGE_END}）')
    parser.add_argument('--n_clusters', type=int, default=DEFAULT_N_CLUSTERS,
                        help=f'手动模式下的聚类数（默认 {DEFAULT_N_CLUSTERS}）')
    parser.add_argument('--input_base', type=str, default=None,
                        help='指定输入 base 目录（如 20260409_131912_base_embeddings），不指定则自动使用最新')
    parser.add_argument('--random_state', type=int, default=DEFAULT_RANDOM_STATE,
                        help=f'随机种子（默认 {DEFAULT_RANDOM_STATE}）')
    parser.add_argument('--n_init', type=int, default=10,
                        help='KMeans 的 n_init 参数（默认 10）')
    parser.add_argument('--max_iter', type=int, default=300,
                        help='KMeans 的最大迭代次数（默认 300）')
    
    # 自动选 K 优化选项
    parser.add_argument('--parallel', action='store_true', default=True,
                        help='启用并行计算（默认启用）')
    parser.add_argument('--no-parallel', dest='parallel', action='store_false',
                        help='禁用并行计算')
    parser.add_argument('--use_minibatch', action='store_true', default=True,
                        help='使用 MiniBatchKMeans 加速评估（默认启用）')
    parser.add_argument('--no_minibatch', dest='use_minibatch', action='store_false',
                        help='使用标准 KMeans')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'MiniBatchKMeans 的批大小（默认 {DEFAULT_BATCH_SIZE}）')
    parser.add_argument('--n_jobs', type=int, default=None,
                        help='并行进程数，默认为 CPU 核心数')
    
    args = parser.parse_args()
    
    # 确定输入目录
    if args.input_base is None:
        base_dir = get_latest_base_dir()
        if base_dir is None:
            print("错误：未找到任何 _base_embeddings 目录，请先运行 02_generate_embeddings.py")
            sys.exit(1)
        print(f"自动使用最新 base 目录: {base_dir.name}")
    else:
        base_dir = Path("outputs") / args.input_base
        if not base_dir.exists():
            print(f"错误：目录 {base_dir} 不存在")
            sys.exit(1)
    
    # 加载降维后的向量
    pca_path = base_dir / "embeddings_pca.npy"
    if not pca_path.exists():
        print(f"错误：{pca_path} 不存在，请先运行 02_generate_embeddings.py")
        sys.exit(1)
    X = np.load(pca_path)
    print(f"加载向量形状: {X.shape}")
    
    # 确定聚类数 K
    if args.auto_k:
        if not KNEE_AVAILABLE:
            print("错误：自动选 K 需要安装 kneed 库，请运行 'pip install kneed'")
            sys.exit(1)
        optimal_k, inertias = find_optimal_k_by_elbow(
            X,
            k_range_start=args.k_range_start,
            k_range_end=args.k_range_end,
            random_state=args.random_state,
            n_init=args.n_init,
            max_iter=args.max_iter,
            use_minibatch=args.use_minibatch,
            batch_size=args.batch_size,
            parallel=args.parallel,
            n_jobs=args.n_jobs
        )
        n_clusters = optimal_k
        print(f"\n自动确定的最佳聚类数 K = {n_clusters}")
        # 保存 inertia 曲线数据到 CSV
        k_values = list(range(args.k_range_start, args.k_range_end + 1))
        inertia_df = pd.DataFrame({"K": k_values, "inertia": inertias})
        inertia_csv = base_dir / "auto_k_inertia.csv"
        inertia_df.to_csv(inertia_csv, index=False)
        print(f"inertia 曲线数据已保存到 {inertia_csv}")
    else:
        n_clusters = args.n_clusters
        print(f"\n使用手动指定的聚类数 K = {n_clusters}")
    
    # 执行最终 K-means 聚类（使用标准 KMeans 以获得最佳质量）
    print(f"执行最终 K-means 聚类，簇数 = {n_clusters} ...")
    final_kmeans = KMeans(n_clusters=n_clusters,
                          random_state=args.random_state,
                          n_init=args.n_init,
                          max_iter=args.max_iter,
                          verbose=0)
    labels = final_kmeans.fit_predict(X)
    print("聚类完成")
    
    # 创建输出目录（基于 base_dir 名称，加上 K 值后缀）
    output_dir_name = base_dir.name + f"_kmeans_K{n_clusters}"
    output_dir = base_dir.parent / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存标签
    np.save(output_dir / "cluster_labels.npy", labels)
    
    # 读取原始文本（从 base_dir 对应的原始数据目录获取）
    raw_timestamp = base_dir.name.replace("_base_embeddings", "")
    raw_dir = base_dir.parent / raw_timestamp
    txt_path = raw_dir / "cleaned_intents.txt"
    if not txt_path.exists():
        print(f"错误：找不到原始文本文件 {txt_path}")
        sys.exit(1)
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
    # 多进程并行必须在 __main__ 块中执行
    main()


'''
# 默认：并行 + MiniBatch，自动搜索 K=2~500
python scripts/03_cluster_kmeans_miniBatchKmeans.py --auto_k

# 缩小搜索范围，使用 8 个进程
python scripts/03_cluster_kmeans_miniBatchKmeans.py --auto_k --k_range_end 300 --n_jobs 8

# 禁用并行（串行），但仍使用 MiniBatch
python scripts/03_cluster_kmeans_miniBatchKmeans.py --auto_k --no-parallel

# 使用标准 KMeans（不推荐，除非需要精确 inertia 对比）
python scripts/03_cluster_kmeans_miniBatchKmeans.py --auto_k --no_minibatch

# 手动指定 K 值（与原有行为一致）
python scripts/03_cluster_kmeans_miniBatchKmeans.py --n_clusters 150
'''