#!/usr/bin/env python3
import subprocess
import sys
from datetime import datetime

# 1. 清洗
print("=== 步骤1: 清洗数据 ===")
subprocess.run([sys.executable, "scripts/01_clean_with_data_juicer.py"])

# 获取最新时间戳（从清洗输出中解析，或简单等待）
# 这里简化：直接运行聚类脚本（它会自动找最新）
print("=== 步骤2: 聚类 ===")
subprocess.run([sys.executable, "scripts/02_cluster_with_text2vec_kmeans.py"])

print("=== 步骤3: 分析与可视化 ===")
subprocess.run([sys.executable, "scripts/03_analyze_kmeans.py"])