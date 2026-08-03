# 作业三：机器学习入门代码实践

本工程是第二部分《机器学习入门知识总结》的代码实践。为了适合我使用 VS Code、
Python 插件和 `uv` 持续维护，我选择了“完整 Python 工程”形式，而不是把全部内容放进
若干 Notebook。运行一次主命令后，程序会训练所有模型、计算评价指标、使用 Matplotlib
保存图片，并自动更新 [RESULTS.md](RESULTS.md)。

工程对应的知识库为
[AI-Knowledge-Base](https://git.ccysam.xyz/ccysam/AI-Knowledge-Base)，实现范围以其中
Machine Learning 的 1、2、3 和第二部分报告已经总结的深度学习模型为准。

## 当前状态

- 作业对应：机器学习入门阶段作业第三部分。
- 完成情况：13 个模型及评价流程均已实现，7 项自动化测试通过。
- 实验产出：结构化指标、结果报告和 11 张 Matplotlib 图片均已生成。
- 维护状态：核心内容已完成，后续继续补充注释、实验和学习记录。

## 一、完成内容

| 部分 | 模型或实践 | 实现方式 | 数据 |
|---|---|---|---|
| Machine Learning 1 | 线性回归 | NumPy 手写批量梯度下降 | 合成回归数据 |
| Machine Learning 1 | 逻辑回归 | NumPy 手写 Sigmoid、交叉熵和梯度下降 | 乳腺癌二分类 |
| Machine Learning 1 | 决策树 | scikit-learn，并绘制树结构 | 乳腺癌二分类 |
| Machine Learning 1 | 高斯朴素贝叶斯 | NumPy 手写先验与类条件高斯分布 | 乳腺癌二分类 |
| Machine Learning 2 | 支持向量机 | 标准化 + RBF 核 SVM | 乳腺癌二分类 |
| Machine Learning 2 | 随机森林 | Bagging 与随机特征子集 | 乳腺癌二分类 |
| Machine Learning 2 | 梯度提升 | 顺序拟合弱学习器 | 乳腺癌二分类 |
| Machine Learning 2 | K-Means | NumPy 手写 K-Means++ 和交替迭代 | 三簇合成数据 |
| Machine Learning 2 | 层次聚类 | Ward 凝聚聚类与树状图 | 三簇合成数据 |
| Machine Learning 2 | PCA | NumPy 手写协方差矩阵特征分解 | Wine 数据集 |
| Machine Learning 3 | 优化与评价 | 特征缩放、五折交叉验证、网格搜索、多种指标 | 乳腺癌二分类 |
| Deep Learning | MLP | PyTorch 全连接网络 | 8×8 手写数字 |
| Deep Learning | CNN | PyTorch 卷积、ReLU 与最大池化 | 8×8 手写数字 |
| Deep Learning | Self-Attention | PyTorch 显式实现 `softmax(QKᵀ/√dₖ)V` | 将图像的 8 行视作 8 个 token |

线性回归、逻辑回归、高斯朴素贝叶斯、K-Means 和 PCA 使用 NumPy 手写，便于将数学公式
与代码逐行对应。决策树、SVM 和集成模型使用 scikit-learn，重点放在规范训练、评价和比较；
三个深度学习模型使用 PyTorch，展示前向传播、损失、反向传播和参数更新。

## 二、工程结构

```text
machine-learning-practice/
├─ .python-version                 # uv 使用 Python 3.12
├─ .vscode/                        # VS Code 解释器与测试配置
├─ pyproject.toml                  # 依赖、命令入口和 pytest 配置
├─ uv.lock                         # uv sync 后生成的依赖锁文件
├─ README.md                       # 安装、运行与 Matplotlib 入门
├─ RESULTS.md                      # 自动生成的实验结果说明
├─ src/ml_practice/
│  ├─ from_scratch.py              # 五个 NumPy 手写算法
│  ├─ classical.py                 # 经典机器学习实验
│  ├─ deep_learning.py             # MLP、CNN、自注意力
│  ├─ plotting.py                  # Matplotlib 公共绘图函数
│  └─ run_all.py                   # 全部实验的命令行入口
├─ tests/                          # 数学实现与网络结构测试
└─ results/
   ├─ metrics.json                 # 结构化指标
   ├─ metrics.csv                  # 可用 Excel 打开的指标
   └─ figures/                     # 11 张 PNG 结果图
```

## 三、第一次运行：uv + VS Code

### 1. 在 VS Code 中打开工程

在 Git Bash 中执行：

```bash
cd /d/rpc_work/Rpc_-Learning_Journey/2026-summer/projects/machine-learning-practice
code .
```

VS Code 提示推荐扩展时，安装 Microsoft 发布的 **Python** 和 **Pylance**。如果没有提示，
在扩展商店中搜索这两个名称安装即可。本作业是普通 Python 工程，不需要安装 Jupyter 扩展。

### 2. 创建虚拟环境并安装依赖

在 VS Code 集成终端中执行：

```bash
uv sync
```

`uv` 会读取 `.python-version` 和 `pyproject.toml`，自动准备 Python 3.12、创建 `.venv`
并安装 NumPy、SciPy、scikit-learn、Matplotlib、PyTorch 和 pytest。不要再手动运行
`python -m venv`，也不要在这个工程中混用 `pip install`。

### 3. 让 VS Code 使用项目环境

按 `Ctrl+Shift+P`，输入并选择 `Python: Select Interpreter`，然后选择：

```text
.venv/Scripts/python.exe
```

状态栏显示 `.venv` 或 Python 3.12 后，编辑器补全、运行按钮和测试功能都会使用这个环境。

### 4. 运行全部实验

```bash
uv run ml-practice
```

等价的模块命令是：

```bash
uv run python -m ml_practice.run_all
```

命令将按顺序运行经典机器学习、MLP、CNN 和自注意力，最后生成或更新：

- `RESULTS.md`
- `results/metrics.json`
- `results/metrics.csv`
- `results/figures/*.png`

### 5. 运行测试

```bash
uv run pytest
```

测试用于检查手写算法是否学到预期规律，以及三个神经网络和注意力矩阵的形状是否正确。

## 四、常用运行选项

只检查经典机器学习和 Matplotlib 环境，不训练神经网络：

```bash
uv run ml-practice --skip-deep-learning
```

先用 3 轮快速检查深度学习代码：

```bash
uv run ml-practice --epochs 3
```

强制使用 CPU，便于不同电脑复现：

```bash
uv run ml-practice --device cpu
```

电脑已经有 CUDA 版 PyTorch 且 CUDA 可用时，可以指定：

```bash
uv run ml-practice --device cuda
```

默认值是 `--device auto`，程序会自动判断；本工程的数据很小，使用 CPU 也可以完成。

## 五、Matplotlib 入门

### 1. 如何安装

本工程已经在 `pyproject.toml` 中声明了 Matplotlib，因此执行 `uv sync` 时会自动安装，
不需要另装。如果以后在其他 uv 工程中单独添加它，可以运行：

```bash
uv add matplotlib
```

### 2. 最小绘图过程

Matplotlib 最重要的四个对象或步骤是：

1. `fig`：整张画布；
2. `ax`：画布上的一个坐标区域；
3. `ax.plot`、`ax.scatter`、`ax.bar`、`ax.imshow`：折线、散点、柱状图和图像；
4. `fig.savefig`：把画布保存成文件。

最小示例如下：

```python
from pathlib import Path

import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y = [0.8, 0.86, 0.91, 0.93]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, y, marker="o", label="test accuracy")
ax.set_title("Accuracy after each epoch")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend()
fig.tight_layout()
fig.savefig(Path("results/figures/example.png"), dpi=180)
plt.close(fig)
```

本工程在 `plotting.py` 中设置了 `Agg` 后端，所以运行脚本时不会弹出绘图窗口，而会直接
保存 PNG。这正适合“完整工程 + 单独保存图片”的作业形式。`plt.close(fig)` 用来释放画布，
批量生成图片时应当保留。

### 3. 本工程的图片分别说明什么

- `01_linear_regression.png`：拟合直线和梯度下降损失；
- `02_supervised_model_comparison.png`：六个分类模型的 Accuracy、F1、ROC-AUC；
- `03_decision_tree_structure.png`：决策树的划分节点；
- `04_ensemble_models.png`：随机森林特征重要性和梯度提升阶段结果；
- `05_cross_validation_and_tuning.png`：五折交叉验证；
- `06_unsupervised_learning.png`：K-Means、层次聚类树状图和 PCA；
- `07_deep_learning_curves.png`：MLP、CNN、自注意力的损失与准确率；
- `08_deep_learning_confusion_matrices.png`：三个神经网络的混淆矩阵；
- `09_cnn_sample_predictions.png`：CNN 对若干手写数字的预测；
- `10_self_attention_heatmap.png`：图像各行之间的注意力权重；
- `11_result_summary.png`：按任务分面的最终统计图。

## 六、如何阅读和扩展代码

建议先阅读 `from_scratch.py` 中的线性回归和逻辑回归，把损失函数、梯度和更新公式与
第二部分报告对照；随后阅读 K-Means 的“分配—更新—收敛”和 PCA 的特征分解；再阅读
`classical.py` 中统一的数据划分与指标。最后阅读 `deep_learning.py` 的 `_train` 函数，
观察 PyTorch 中完整的训练循环：

```text
optimizer.zero_grad()
→ logits = model(features)       # 前向传播
→ loss = criterion(logits, y)    # 计算损失
→ loss.backward()                # 反向传播
→ optimizer.step()               # 更新参数
```

新增模型时，可以在相应实验模块中训练并返回一个指标字典，再在 `run_all.py` 中把它加入
汇总。所有数据集均来自 scikit-learn 本地包，实验阶段不依赖网络下载。

## 七、许可

本项目源代码采用仓库根目录 [LICENSE](../../../LICENSE) 中的 MIT License；README、
自动生成的结果报告和原创结果图适用同一文件中说明的 CC BY-NC-SA 4.0。第三方依赖和
数据集仍遵循其各自的许可。
