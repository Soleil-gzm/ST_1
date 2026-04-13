# sentence-transformers 意图聚类项目

## 项目目标
基于 `text2vec` 和 K‑means 的用户意图自动分组，用于将海量意图文本（约 12 万条）聚类成可管理的主题簇（如 500‑1500 个），便于后续业务分析和模型训练。

## 项目结构

```ST_1/
├── data/
│   └── user_intents.json                		# 原始意图数据（仅含 user_intents 字段）
├── configs/
│   └── data_juicer_clear_config.yaml    		# Data‑Juicer 清洗配置
├── outputs/                             		# 所有输出（按时间戳组织）
│   ├── 20260407_174104/                 		# 清洗后文本目录
│   │   └── cleaned_intents.txt
│   └── 20260407_174104_text2vec_kmeans/ 		# 聚类结果目录
│       ├── embeddings.npy               		# 原始 768 维向量
│       ├── embeddings_pca.npy           		# PCA 降维后向量（用于聚类）
│       ├── cluster_labels.npy           		# 聚类标签
│       ├── clustered_intents.csv        		# 带标签的意图 CSV
│       ├── cluster_summary.csv          		# 簇汇总（大小、代表性文本）
│       ├── cluster_view.html            		# 可折叠簇视图（HTML）
│       ├── cluster_visualization.png    		# 静态 UMAP 散点图
│       └── cluster_visualization_interactive.html  		# 交互式散点图
├── scripts/
│   ├── 01_clean_with_data_juicer.py    		# Data‑Juicer 清洗
│   ├── 02_generate_embeddings.py       		# 生成向量 + PCA 降维
│   ├── 03_cluster_kmeans.py            		# K‑means 聚类（可调 K 值）
│   ├── 04_analyze_kmeans.py            		# 分析 + 可视化 + 簇视图
│   ├── evaluate_kmeans_elbow.py        		# 肘部法则评估最优 K
│   └── run_kmeans_pipeline.py          		# 一键运行全流程
├── new_sentence_transformers_env.yaml  		# 聚类环境导出
├── data-juicer_env.yaml                		# 清洗环境导出
└── README.md
```

### 1. 环境配置

项目依赖两个 conda 环境：

>     data-juicer：用于数据清洗        
>     new_sentence_transformers：用于向量化 + 聚类（推荐新建）


创建并激活聚类环境：

    >     bash
    >     conda env create -f new_sentence_transformers_env.yaml
    >     conda activate new_sentence_transformers

若需手动安装核心依赖：

    >     bash
    >     pip install sentence-transformers scikit-learn pandas numpy matplotlib umap-learn plotly

### 2. 准备数据

将原始意图 JSON 文件放入 `data/user_intents.json`，格式：

```json

{
 "user_intents": [
 "查询信用卡账单",
 "如何申请分期付款",
 "投诉客服态度差",
 ...
 ]
}
```
### 3. 运行全流程（推荐）

    >     bash
    >     enter code herepython scripts/run_kmeans_pipeline.py

该脚本依次执行清洗 → 生成向量 → 聚类 → 分析，最终结果保存在 `outputs/` 下最新时间戳目录中。
### 4. 分步运行（便于调试）

    bash
    
    # 步骤1：清洗数据（使用 Data‑Juicer）
    python scripts/01_clean_with_data_juicer.py
    # 步骤2：生成向量 + PCA 降维（只需一次）
    python scripts/02_generate_embeddings.py
    # 步骤3：K‑means 聚类（可反复调整 K 值）
    # 修改 03_cluster_kmeans.py 中的 N_CLUSTERS 变量
    python scripts/03_cluster_kmeans.py
    # 步骤4：分析并生成可视化报告
    python scripts/04_analyze_kmeans.py
## 调整聚类数量（K 值）

当前默认 K=1000。若需找到最佳 K，可使用肘部法则：

    >     bash
    >     python scripts/evaluate_kmeans_elbow.py

该脚本会测试预设的 K 值列表（可自行修改），生成 `kmeans_elbow.png` 曲线图，帮助选择拐点处的 K 值。

修改后的03聚类脚本结合了自动寻找K值的功能，需要在命令行运行：

    > python scripts/03_cluster_kmeans.py --auto_k

正常运行还是和原来一样，使用手动设置。


## 常见问题

**Q1：为什么清洗后意图数量减少很多？**  
A：Data‑Juicer 配置中包含了长度、语言、停用词等过滤器，会剔除低质量或无效意图。可根据需求调整 `configs/data_juicer_clear_config.yaml` 中的阈值。

**Q2：向量生成速度慢 / 内存不足？**  
A：在 `02_generate_embeddings.py` 中减小 `batch_size`（如 128）或降低 `PCA_COMPONENTS`（如 50）。

**Q3：聚类后一个簇过大，其他簇很小？**  
A：可能 K 值太小或数据分布不均。尝试增大 K 值，或使用 HDBSCAN 自动发现簇（效果不是很好）。

**Q4：如何对比不同 K 值的聚类效果？**  
A：每次修改 `N_CLUSTERS` 后重新运行 `03_cluster_kmeans.py`，结果会保存在带 K 值的独立目录中（如 `..._kmeans_K800`），便于对比。

**Q5：如何找到最佳K 值的聚类参数？**  
A：运行脚本`evaluate_kmeans_elbow.py`，它会自动找到最新的 `_base_embeddings` 目录，并将生成的 CSV 和 PNG 文件保存在该目录下。

## 技术栈

>     清洗：Data‑Juicer（规则 + 启发式过滤）
>     向量化：`text2vec-base-chinese`（Sentence‑Transformer） 
>     降维：PCA（主成分分析）
>     聚类：K‑means + 肘部法则
>     可视化：UMAP + Matplotlib + Plotly
## 许可

本项目仅用于内部数据分析，不对外开源。如有疑问请联系项目维护者。