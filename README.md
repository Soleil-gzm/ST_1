# sentence-transformers 意图聚类项目

## 项目目标
对约12万条用户意图文本进行自动分组，得到3000-4000个语义簇，便于后续分析和模型训练。

## 工作流程
1. **清洗**：使用 Data-Juicer 去除无效、重复、无意义的意图。
2. **向量化**：使用 Sentence-Transformers 将文本转换为语义向量。
3. **降维与聚类**：使用 UMAP 降维 + HDBSCAN 聚类，自动确定簇数量。
4. **分析与可视化**：生成簇统计报告和散点图。

## 环境要求
- Python 3.8+
- 建议使用 GPU（可选，CPU 也可运行但较慢）

## 安装
```bash
pip install -r requirements.txt

## 项目结构
intent_clustering/
├── data/
│   └── raw_intents.json               # 原始意图数据（仅含 user_intents 字段）
├── configs/
│   └── data_juicer_config.yaml        # Data-Juicer 清洗配置
├── outputs/
│   ├── cleaned_intents.txt            # 清洗后的意图（每行一个）
│   ├── embeddings.npy                 # 向量矩阵
│   ├── reduced_embeddings.npy         # 降维后的向量
│   ├── cluster_labels.npy             # 聚类标签
│   ├── clustered_intents.csv          # 带标签的意图（CSV）
│   ├── cluster_summary.csv            # 簇汇总（大小、代表性文本）
│   └── cluster_visualization.png      # 可视化散点图
├── scripts/
│   ├── 01_clean_with_data_juicer.py   # 调用 Data-Juicer 清洗
│   ├── 02_embed.py                    # 向量化
│   ├── 03_cluster.py                  # 降维 + 聚类
│   └── 04_analyze.py                  # 分析并生成报告
├── requirements.txt
├── run_all.sh                         # 一键运行所有脚本
└── README.md