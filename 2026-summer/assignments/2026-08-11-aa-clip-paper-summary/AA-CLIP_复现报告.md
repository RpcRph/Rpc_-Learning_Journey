# AA-CLIP 代码复现报告

> 论文：*AA-CLIP: Enhancing Zero-Shot Anomaly Detection via Anomaly-Aware CLIP*（CVPR 2025）  
> 论文地址：https://arxiv.org/abs/2503.06661  
> 官方代码：https://github.com/Mwxinnn/AA-CLIP  
> 复现日期：2026-08-11；续跑日期：2026-08-15 至 2026-08-16  
> 复现平台：AutoDL，单张 NVIDIA GeForce RTX 3090 24 GB

## 1. 复现目标与结论概览

本次复现不是只执行一次官方命令，而是完成一条可检查、可恢复、可重复执行的实验链路：

1. 从 Windows 本地配置独立 SSH 密钥并连接 AutoDL；
2. 审计远端 GPU、驱动、CUDA、磁盘和 Python 环境；
3. 固定官方仓库版本并修复安装、数据路径和新依赖兼容问题；
4. 下载并校验 OpenAI CLIP ViT-L/14@336 权重；
5. 下载、上传、解压并逐条校验 VisA 与 MVTec AD；
6. 生成有固定随机种子和哈希值的 2/16/64-shot 训练清单；
7. 在 VisA 上训练 AA-CLIP，并在未见过的 MVTec AD 上执行 zero-shot 评估；
8. 将实际指标与论文表格进行比较，并记录不能逐位复原论文数字的原因。

截至本报告当前版本，用户要求优先完成的两组核心协议均已结束：

- VisA 2/16/64/full-shot → MVTec 全部完成；
- MVTec 2/16/64/full-shot → VisA 全部完成；
- 共生成并核对 8 个 results.csv；新增 5 个运行目录和总流水线日志均无 Traceback、OOM、RuntimeError 或 Error；
- VisA full-shot → MVTec（epoch 15）得到 pixel AUROC 92.1260、image AUROC 89.4393，与论文分别相差 +0.2260 和 -1.0607 个百分点；
- MVTec → VisA 的 2/16/64/full-shot pixel AUROC 分别为 94.5375、94.5092、94.7408、94.8683；
- 对应 image AUROC 分别为 76.8608、80.4483、81.8583、76.8017；前三档呈改善趋势，但 full-shot 图像级结果低于论文 7.7983 个百分点；
- 顺序流水线于 2026-08-16 04:01:06 完成，GPU 已空闲，所有 checkpoint、训练日志、测试日志、CSV 和哈希均保存在 AutoDL 数据盘。

## 2. 论文实验协议核对

论文实验设置中与本次复现直接相关的内容如下：

- 骨干网络：OpenCLIP/CLIP ViT-L/14；
- 输入尺寸：518×518；
- 多层视觉特征：第 6、12、18、24 层；
- 文本适配层数：`K_T = 3`；
- 图像适配层数：`K_I = 6`；
- 残差权重：`lambda = 0.1`；
- 解耦损失权重：`gamma = 0.1`；
- 阶段一：文本适配器训练 5 epoch，学习率 `1e-5`；
- 阶段二：图像适配器训练 20 epoch，学习率 `5e-4`；
- 优化器：Adam；
- 硬件：单张 NVIDIA GeForce RTX 3090；
- few-shot 规模：每类 2、16、64 个样本，正常与异常保持 1:1；
- 主要协议：在 VisA 上训练，在 MVTec、BTAD、MPDD 和医学数据集上进行跨数据集 zero-shot 测试；
- 论文中的 VisA 指标则反过来使用 MVTec 作为训练数据。

因此，本次“VisA 训练、MVTec 测试”属于论文规定的正式 zero-shot 协议；“VisA 训练、VisA 测试”只能用于验证程序链路，不能作为论文指标。

## 3. 从本地连接 AutoDL

### 3.1 创建独立 SSH 密钥

为了不在自动化过程中使用实例密码，本次使用了单独的 RSA 密钥：

```powershell
ssh-keygen -t rsa -b 4096 -f "$env:USERPROFILE\.ssh\id_rsa_autodl_aaclip"
Get-Content "$env:USERPROFILE\.ssh\id_rsa_autodl_aaclip.pub"
```

将公钥添加到 AutoDL 容器的 `root` 用户后，在本地
`C:\Users\Rpc\.ssh\config` 中配置别名：

```sshconfig
Host autodl-aaclip
    HostName connect.nmb2.seetacloud.com
    User root
    Port 32489
    IdentityFile ~/.ssh/id_rsa_autodl_aaclip
    IdentitiesOnly yes
    BatchMode yes
    ServerAliveInterval 30
    ServerAliveCountMax 6
```

后续连接只需要：

```powershell
ssh autodl-aaclip
```

文件上传也可以直接使用该别名：

```powershell
scp local-file autodl-aaclip:/root/autodl-tmp/
```


## 4. AutoDL 实例审计

连接成功后执行：

```bash
hostname
nvidia-smi
df -h /root/autodl-tmp
which python
```

实际环境：

| 项目 | 实际值 |
|---|---|
| 主机名 | `autodl-container-32304ab2ca-99435ad2` |
| GPU | NVIDIA GeForce RTX 3090 |
| 显存 | 24,576 MiB |
| NVIDIA 驱动 | 570.124.04 |
| 系统 CUDA | 12.1 |
| 数据盘 | `/root/autodl-tmp`，总计 50 GB |
| Conda | `/root/miniconda3/bin/conda` |

AutoDL 非交互 SSH 的 `PATH` 中没有 Conda，因此脚本中没有依赖 `conda activate`，而是始终使用绝对 Python 路径：

```text
/root/autodl-tmp/envs/aaclip/bin/python
```

这样可以避免交互终端能运行、后台 `screen` 或 SSH 命令却找不到环境的问题。

## 5. 固定官方代码版本

仓库放在数据盘而不是容量较小的系统盘：

```bash
cd /root/autodl-tmp
git clone https://github.com/Mwxinnn/AA-CLIP.git
cd AA-CLIP
git fetch --depth 1 origin main
git checkout -B codex/reproduction FETCH_HEAD
```

本次固定的官方提交为：

```text
53db195f230442aa118c246876c94ba1c76139cc
```

远端项目路径：

```text
/root/autodl-tmp/AA-CLIP
```

固定提交号很重要，因为仓库之后若继续更新，默认分支上的行为、依赖或测试方式可能改变。

## 6. Python/CUDA 环境安装

### 6.1 最终环境

创建 Python 3.10 环境：

```bash
/root/miniconda3/bin/conda create \
  --prefix /root/autodl-tmp/envs/aaclip \
  --override-channels \
  --channel https://repo.anaconda.com/pkgs/main \
  python=3.10 pip -y
```

最终关键版本：

| 包或运行库 | 版本 |
|---|---|
| Python | 3.10 |
| PyTorch | 2.3.1+cu121 |
| torchvision | 0.18.1+cu121 |
| CUDA runtime | 12.1 |
| cuDNN | 8.9.0 |
| NCCL | 2.20.5 |
| kornia | 0.6.9 |
| pandas | 2.2.2 |
| scikit-learn | 1.5.0 |

环境验证输出：

```text
torch=2.3.1+cu121
torchvision=0.18.1+cu121
kornia=0.6.9
pandas=2.2.2
sklearn=1.5.0
cuda_runtime=12.1
cuda_available=True
cudnn=8900
gpu=NVIDIA GeForce RTX 3090
cuda_sum=1048576.0
allocated_mib=4.0
```

### 6.2 遇到的问题：官方 requirements.txt 不能直接安装

官方 `requirements.txt` 实际上是 Conda 导出表，包含包名、版本、build 和 channel 等多列，不是合法的 pip requirements 格式。例如：

```text
einops 0.7.0 pypi_0 pypi
```

解决方法是整理为真正的固定版本依赖文件，并为 AutoDL 单独提供 `requirements-autodl.txt`。

### 6.3 遇到的问题：Conda 镜像和 PyTorch CUDA 依赖

默认 Conda 镜像出现访问问题，因此创建环境时使用 `--override-channels` 指向官方 channel。

直接安装 PyTorch 及其全部 CUDA wheel 会重复下载实例中已经存在的 CUDA 组件，下载量接近数 GB。解决方法是：

1. 从 PyTorch CUDA 12.1 索引安装 `torch==2.3.1+cu121` 和 `torchvision==0.18.1+cu121`；
2. 对已由系统 CUDA 提供的部分使用 `--no-deps`，避免重复下载；
3. 其他 Python 包通过固定的 `requirements-autodl.txt` 安装。

### 6.4 遇到的问题：NCCL 符号不匹配

第一次导入 PyTorch 时出现：

```text
undefined symbol: ncclCommRegister
```

原因是系统 NCCL 与 PyTorch 2.3.1 期望的版本不匹配。单独安装：

```bash
pip install --no-deps nvidia-nccl-cu12==2.20.5
```

之后 CUDA 张量测试成功。

## 7. 官方代码兼容性修复

本次默认保留官方模型结构与损失逻辑，只修复会阻止安装、启动、数据加载或结果汇总的问题。

### 7.1 删除作者机器的硬编码数据路径

官方 `dataset/constants.py` 使用作者本地路径：

```python
BASE_PATH = "/data/wenxinma"
```

改为默认使用仓库 `data/`，也允许环境变量覆盖：

```python
BASE_PATH = os.environ.get(
    "AACLIP_DATA_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")),
)
```

随后还修复了一次 `data/data/...` 重复拼接问题，使 VisA 和 MVTec 分别解析为：

```text
/root/autodl-tmp/AA-CLIP/data/VisA_20220922
/root/autodl-tmp/AA-CLIP/data/mvtec_ad
```

### 7.2 scripts 目录启动时找不到 model 包

直接执行：

```bash
python scripts/smoke_model.py
```

最初报错：

```text
ModuleNotFoundError: No module named 'model'
```

原因是脚本目录而不是仓库根目录成为 `sys.path[0]`。解决方法是在脚本中根据 `__file__` 找到仓库根目录并加入 `sys.path`。

### 7.3 tokenizer 使用过时的 pkg_resources 接口

第一次正式训练报错：

```text
ImportError: cannot import name 'packaging' from 'pkg_resources'
```

官方代码：

```python
from pkg_resources import packaging
```

修复为标准接口：

```python
from packaging import version
```

并将 `packaging==24.1` 明确加入依赖。

### 7.4 pandas 2.x 无法直接对混合类型 DataFrame 求平均

第一次完整测试在 12 类推理全部结束后报错：

```text
TypeError: Could not convert [...] to numeric
```

原因是官方代码调用：

```python
df.mean()
```

pandas 2.x 不再默认忽略字符串类别列。第一次改成 `numeric_only=True` 后，由于四个指标列最初也是 `object` 类型，得到的平均行仍是 NaN。

最终采用显式数值转换：

```python
metric_columns = ["pixel AUC", "pixel AP", "image AUC", "image AP"]
average_row = {"class name": "Average"}
average_row.update(df[metric_columns].astype(float).mean().to_dict())
df.loc[len(df)] = Series(average_row)
```

该改动只影响结果表汇总，不改变模型预测。

### 7.5 增加指定 checkpoint 评估

官方 `test.py` 会遍历目录中的全部 20 个 checkpoint，单次完整测试成本很高。新增参数：

```bash
--checkpoint_epochs 15 20
```

不传该参数时仍保持官方行为；传入后只评估指定 epoch。

## 8. CLIP 权重下载与模型冒烟测试

官方模型要求文件名：

```text
model/ViT-L-14-336px.pt
```

使用 OpenAI 官方权重：

```text
https://openaipublic.azureedge.net/clip/models/
3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/
ViT-L-14-336px.pt
```

期望 SHA-256：

```text
3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02
```

下载脚本支持断点续传，只有哈希一致才会把 `.part` 文件改为正式权重。

518×518 前向测试输出：

```text
gpu=NVIDIA GeForce RTX 3090
image_feature_shape=(1, 768)
patch_feature_shapes=[
  (1, 1370, 1024),
  (1, 1370, 1024),
  (1, 1370, 1024),
  (1, 1370, 1024)
]
peak_vram_mib=3275.6
```

这证明权重、位置编码调整、DPAM 替换、CUDA 和模型多层输出均可工作。

## 9. 数据准备与校验

### 9.1 VisA

官方数据地址：

```text
https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar
```

数据包信息：

| 项目 | 值 |
|---|---|
| 文件大小 | 1,929,840,640 bytes |
| SHA-256 | `2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362` |
| 解压后大小 | 约 1.9 GB |

AutoDL 到 Amazon S3 的单连接速度一度只有约 0.3–0.7 MB/s，因此下载脚本实现了 8 个 HTTP Range 并行分段。中间还出现过旧单连接 curl 在 SSH 中成为孤儿进程并继续写 `.part` 的情况。处理步骤是：

1. 确认具体旧进程 PID；
2. 只终止该 curl 进程；
3. 根据第一个并行 range 的起点，将前缀文件精确截断回 45,363,200 bytes；
4. 完成 8 段下载并重组；
5. 检查总大小并完整扫描 tar。

VisA tar 没有 `VisA_20220922/` 顶层目录，而是直接包含 `candle/`、`pcb1/` 等类别目录。最初按预期顶层目录解压后，脚本报告目标目录不存在。解决方法是直接把 tar 解压到预先建立的 `data/VisA_20220922/`。

官方 AA-CLIP metadata 共引用 2,162 条 VisA 记录。逐条检查所有图像和异常 mask 后：

```text
all_referenced_files_exist=True
```

### 9.2 MVTec AD

MVTec AD 需要从官方页面接受许可后下载。用户下载到本地：

```text
D:\datasets\anomaly\mvtec_anomaly_detection.tar
```

本地文件信息：

| 项目 | 值 |
|---|---|
| 文件大小 | 5,276,467,200 bytes |
| SHA-256 | `689963d5b59a9c0f2b87f7ec379c742acfabc1a99428dff0fa32012a7b137556` |

由于文件约 5.28 GB，采用 SFTP `reput` 上传到：

```text
/root/autodl-tmp/AA-CLIP/data/.downloads/mvtec_anomaly_detection.tar.part
```

`reput` 要求远端续传目标预先存在。第一次因为目标不存在而安全退出，之后先创建空 `.part` 文件，再恢复上传。上传结束后：

1. 远端大小必须等于 5,276,467,200 bytes；
2. 远端重新计算 SHA-256；
3. 哈希完全一致后才改名为正式 tar；
4. 解压到 `data/mvtec_ad/`；
5. 逐条检查 AA-CLIP 的 1,725 条 MVTec metadata。

最终 15 类图像和异常 mask 全部存在：

```text
dataset=MVTec manifest=full-shot.jsonl records=1725
all_referenced_files_exist=True
```

## 10. Few-shot 清单生成

官方仓库只提供 `full-shot.jsonl`，没有发布论文使用的 2/16/64-shot 文件或具体样本索引。

本次采用透明、确定性的替代方案：

1. 按类别和标签分组；
2. 每组候选路径排序；
3. 使用 `random.Random(111)`；
4. 正常和异常各取 shot 的一半；
5. 写入紧凑 JSONL；
6. 记录文件 SHA-256。

| shot | 每类正常 | 每类异常 | 12 类总样本 | 清单 SHA-256 |
|---:|---:|---:|---:|---|
| 2 | 1 | 1 | 24 | `f80976ec18265a69ed83f526b1e52fcd2de63c0dc696fb2a3e0e76156c8c4cc9` |
| 16 | 8 | 8 | 192 | `05a16f3048b2344760823b07fef73a58125154651c145bf521da8b20c384e517` |
| 64 | 32 | 32 | 768 | `79258ae61e01862614295e653196fbe10ec7fa79fcda7865349b454fed34616f` |

这个方案可以让本次结果被准确重复，但不能保证与作者未公开的随机抽样完全相同。

为执行论文中的反向协议（MVTec 训练、VisA 测试），还生成了 MVTec few-shot 清单。官方 `full-shot.jsonl` 只记录 MVTec 测试集，部分类别的 `test/good` 正常图像少于 32 张，因此正常候选池显式合并了 `train/good`。此外，toothbrush 类只有 30 张异常图，无法提供 32 个唯一异常样本；64-shot 清单对其中 2 张异常图各重复一次，以维持每类 32 正常 + 32 异常。这个近似处理会作为复现限制保留，不视为作者未公开协议的精确还原。

| MVTec 清单 | 记录数 | SHA-256 | 说明 |
|---:|---:|---|---|
| 2-shot | 30 | `4570cd6bb0e258687131dd7a937add625839686bbe950bdef4f790d13aaf4441` | 全部唯一 |
| 16-shot | 240 | `362f7874e3f92da0e8a999e97781e01c974db5031163e6d4c8ecb7b8b48afbef` | 全部唯一 |
| 64-shot | 960 | `e9980769de6d1c5461f3b33ee6f27507e6a75e9c37761a1e26eed5d67fd18c8b` | toothbrush 异常重复 2 条 |

## 11. 实验运行与结果

### 11.1 2-shot 训练

后台运行命令：

```bash
screen -dmS aaclip-visa-2shot bash -lc \
  'cd /root/autodl-tmp/AA-CLIP && \
   ./scripts/run_train.sh VisA 2 111 few_shot'
```

运行目录：

```text
runs/VisA_few_shot_2shot_seed111/
```

实际时间：

```text
开始：2026-08-11 18:04:09 +08:00
完成：2026-08-11 18:07:44 +08:00
总计：约 3 分 35 秒
```

损失变化：

| 阶段 | 初始 loss | 最终 loss |
|---|---:|---:|
| 文本适配器，5 epoch | 1.5887 | 1.2782 |
| 图像适配器，20 epoch | 5.1481 | 2.5697 |

资源使用：

- 文本阶段同时保留两个 ViT-L 模型，峰值显存约 21.8 GiB；
- 图像阶段显存约 15.6 GiB；
- 共生成 20 个图像适配器 checkpoint；
- 训练目录约 2.5 GB。

### 11.2 VisA 同数据自测

为了验证完整推理和指标代码，使用 epoch 20 在 VisA 上执行了一次测试：

| Pixel AUROC | Pixel AP | Image AUROC | Image AP |
|---:|---:|---:|---:|
| 93.4317 | 26.5908 | 80.7975 | 84.5867 |

这个结果包含训练样本重叠，只是工程自测，不是论文的 zero-shot VisA 结果。

### 11.3 MVTec 正式 zero-shot 结果

使用 VisA 2-shot 模型，在未见过的 MVTec AD 上测试。

epoch 20 每类结果：

| 类别 | Pixel AUROC | Pixel AP | Image AUROC | Image AP |
|---|---:|---:|---:|---:|
| bottle | 87.87 | 47.87 | 77.30 | 92.93 |
| cable | 75.34 | 9.46 | 49.55 | 65.32 |
| capsule | 92.91 | 17.44 | 81.45 | 95.94 |
| carpet | 99.30 | 69.15 | 99.36 | 99.82 |
| grid | 93.69 | 30.09 | 79.87 | 92.48 |
| hazelnut | 93.74 | 35.73 | 52.93 | 67.00 |
| leather | 98.99 | 33.32 | 99.22 | 99.77 |
| metal_nut | 79.63 | 30.04 | 86.41 | 96.33 |
| pill | 83.25 | 13.94 | 64.38 | 90.59 |
| screw | 97.95 | 26.03 | 64.05 | 84.83 |
| tile | 94.19 | 73.53 | 85.68 | 94.28 |
| transistor | 69.48 | 14.11 | 83.21 | 81.56 |
| toothbrush | 91.44 | 19.39 | 80.00 | 90.48 |
| wood | 96.23 | 60.81 | 95.35 | 98.69 |
| zipper | 96.24 | 50.49 | 75.26 | 92.97 |
| **平均** | **90.0167** | **35.4267** | **78.2680** | **89.5327** |

checkpoint 比较：

| 设置 | Pixel AUROC | Image AUROC |
|---|---:|---:|
| 本次 2-shot epoch 15 | 89.8280 | 77.3413 |
| 本次 2-shot epoch 20 | **90.0167** | **78.2680** |
| 论文 2-shot | 91.0 | 85.9 |
| epoch 20 与论文差值 | -0.9833 | -7.6320 |

结论：

- pixel AUROC 距论文不到 1 个百分点，说明定位链路基本复现；
- image AUROC 差距明显；
- epoch 20 同时优于 epoch 15，因此 full-shot 使用 epoch 15 的说明不能解释当前差距；
- 更可能的原因是论文未公开的 2-shot 样本索引、训练波动和 few-shot checkpoint 选取规则。

### 11.4 16-shot 训练与 MVTec 正式结果

16-shot 使用 12 类共 192 个样本，正常/异常各 96 个，seed 为 111。

```text
运行目录：runs/VisA_few_shot_16shot_seed111/
开始时间：2026-08-11 19:49:15 +08:00
完成时间：2026-08-11 20:11:50 +08:00
训练耗时：22 分 35 秒
清单 SHA-256：05a16f3048b2344760823b07fef73a58125154651c145bf521da8b20c384e517
```

训练收敛情况：

| 阶段 | 初始 loss | 最终 loss |
|---|---:|---:|
| 文本适配器，5 epoch | 1.3869 | 1.1333 |
| 图像适配器，20 epoch | 4.5690 | 2.5200 |

图像阶段 20 个 epoch 的 loss 依次为：

```text
4.56898, 3.49490, 3.12371, 2.89377, 2.74347,
2.69975, 2.67922, 2.66563, 2.64309, 2.58407,
2.65191, 2.60798, 2.59556, 2.55270, 2.57866,
2.55226, 2.52770, 2.56123, 2.52763, 2.51997
```

资源与产物：

- 文本阶段峰值显存约 20.9 GiB，图像阶段约 15.6 GiB；
- 生成 1 个文本适配器和 20 个图像适配器 checkpoint；
- 运行目录大小约 2.5 GB；
- 3090 24 GB 全程未发生 OOM。

使用 epoch 20 在 MVTec AD 上进行正式跨数据集评估。测试从 20:12:35 开始，约在 20:21:47 完成，耗时约 9 分 12 秒。逐类结果如下：

| 类别 | Pixel AUROC | Pixel AP | Image AUROC | Image AP |
|---|---:|---:|---:|---:|
| bottle | 90.70 | 56.22 | 91.83 | 97.74 |
| cable | 83.63 | 23.84 | 85.85 | 92.28 |
| capsule | 94.27 | 22.47 | 91.66 | 98.30 |
| carpet | 99.41 | 73.25 | 100.00 | 100.00 |
| grid | 97.21 | 32.44 | 91.48 | 97.38 |
| hazelnut | 96.98 | 49.52 | 97.89 | 98.92 |
| leather | 98.90 | 34.85 | 100.00 | 100.00 |
| metal_nut | 72.07 | 28.62 | 93.11 | 98.44 |
| pill | 81.90 | 17.28 | 79.32 | 95.66 |
| screw | 98.86 | 36.35 | 85.24 | 93.94 |
| tile | 90.19 | 63.08 | 93.61 | 97.83 |
| transistor | 71.92 | 11.74 | 74.08 | 71.69 |
| toothbrush | 92.67 | 23.04 | 83.61 | 94.58 |
| wood | 97.23 | 61.46 | 99.12 | 99.75 |
| zipper | 96.31 | 46.08 | 88.66 | 96.92 |
| **平均** | **90.8167** | **38.6827** | **90.3640** | **95.5620** |

与论文 16-shot MVTec 结果比较：

| 设置 | Pixel AUROC | Image AUROC |
|---|---:|---:|
| 本次 16-shot epoch 20 | 90.8167 | **90.3640** |
| 论文 16-shot | **91.2** | 89.7 |
| 本次与论文差值 | -0.3833 | +0.6640 |

16-shot 的两项 AUROC 都与论文非常接近：pixel AUROC 相差 0.4 个百分点以内，image AUROC 反而高出 0.6640 个百分点。相比 2-shot，image AUROC 从 78.2680 上升到 90.3640，说明低 shot 场景下未公开的样本索引对图像级结果影响很大；当训练样本增至 16-shot 后，这种敏感性明显降低。

### 11.5 实例重启后的恢复与 64-shot 训练

2026-08-15 重新开启 AutoDL 实例后，没有直接重跑，而是先做恢复核验：

```text
GPU：NVIDIA GeForce RTX 3090 24 GB，空闲
代码 commit：53db195f230442aa118c246876c94ba1c76139cc
数据盘：50 GB，已有数据约 23 GB，可用约 28 GB
已有运行：2-shot、16-shot 权重和日志均存在
后台任务：无遗留 screen
64-shot 清单：768 条，所有图像和 mask 均存在
清单 SHA-256：79258ae61e01862614295e653196fbe10ec7fa79fcda7865349b454fed34616f
```

AutoDL 数据盘内容在实例关机后保留半个月，可以从已完成状态继续，而不需要重新上传 MVTec 或重新安装环境。

64-shot 使用 12 类共 768 个样本，每类 32 个正常和 32 个异常样本，seed 为 111。运行信息：

```text
运行目录：runs/VisA_few_shot_64shot_seed111/
开始时间：2026-08-15 15:13:34 +08:00
完成时间：2026-08-15 16:36:49 +08:00
训练耗时：1 小时 23 分 15 秒
```

训练收敛情况：

| 阶段 | 初始 loss | 最终 loss |
|---|---:|---:|
| 文本适配器，5 epoch | 1.2042 | 1.1166 |
| 图像适配器，20 epoch | 3.5920 | 2.3258 |

文本阶段 5 个 epoch 的 loss：

```text
1.20417, 1.12430, 1.11994, 1.11820, 1.11656
```

图像阶段 20 个 epoch 的 loss：

```text
3.59198, 2.78578, 2.71018, 2.66346, 2.62677,
2.59389, 2.55483, 2.52993, 2.49419, 2.51194,
2.49201, 2.46541, 2.46082, 2.46769, 2.44357,
2.41762, 2.38028, 2.35965, 2.35802, 2.32578
```

epoch 9 和 epoch 13 出现小幅 loss 回升，随后继续下降，属于随机批次训练的平台期波动，没有形成发散。图像 loss 总体下降约 35.3%。

资源与产物：

- 文本阶段显存约 19.5 GiB，图像阶段约 15.3 GiB；
- 生成 1 个文本适配器和 20 个图像适配器 checkpoint；
- 运行目录约 2.5 GB；
- 全程未发生 OOM、CUDA、NCCL 或数据读取错误。

### 11.6 64-shot MVTec 正式结果

使用预先固定的 epoch 20 测试 MVTec，没有根据测试集从 20 个 checkpoint 中事后挑选最好结果。测试从 2026-08-15 16:37:36 开始，最终结果日志于约 16:46:39 写入，耗时约 9 分 3 秒。

| 类别 | Pixel AUROC | Pixel AP | Image AUROC | Image AP |
|---|---:|---:|---:|---:|
| bottle | 91.31 | 59.29 | 95.71 | 98.74 |
| cable | 85.03 | 22.89 | 84.61 | 91.56 |
| capsule | 94.83 | 21.84 | 89.43 | 97.83 |
| carpet | 99.25 | 71.02 | 98.56 | 99.62 |
| grid | 95.95 | 31.21 | 94.57 | 98.30 |
| hazelnut | 97.23 | 55.32 | 92.25 | 96.13 |
| leather | 99.03 | 42.85 | 99.97 | 99.99 |
| metal_nut | 72.53 | 26.29 | 78.84 | 95.15 |
| pill | 84.01 | 23.00 | 82.87 | 96.48 |
| screw | 98.77 | 38.67 | 86.31 | 94.38 |
| tile | 90.87 | 66.17 | 96.43 | 98.61 |
| transistor | 72.42 | 13.58 | 79.71 | 78.73 |
| toothbrush | 95.26 | 42.04 | 94.72 | 98.07 |
| wood | 97.45 | 64.11 | 98.51 | 99.56 |
| zipper | 96.12 | 44.73 | 97.30 | 99.31 |
| **平均** | **91.3373** | **41.5340** | **91.3193** | **96.1640** |

与论文 64-shot MVTec 结果比较：

| 设置 | Pixel AUROC | Image AUROC |
|---|---:|---:|
| 本次 64-shot epoch 20 | 91.3373 | 91.3193 |
| 论文 64-shot | **91.6** | **92.0** |
| 本次与论文差值 | -0.2627 | -0.6807 |

两项 AUROC 与论文均相差不到 0.7 个百分点。考虑到论文没有公布实际 64-shot 文件索引和 checkpoint 选择规则，本结果可以视为对官方代码主结果的高一致度近似复现，但不能表述为逐位一致。

### 11.7 2/16/64-shot 趋势汇总

| shot | 本次 Pixel AUROC | 论文 Pixel AUROC | 差值 | 本次 Image AUROC | 论文 Image AUROC | 差值 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 90.0167 | 91.0 | -0.9833 | 78.2680 | 85.9 | -7.6320 |
| 16 | 90.8167 | 91.2 | -0.3833 | 90.3640 | 89.7 | +0.6640 |
| 64 | 91.3373 | 91.6 | -0.2627 | 91.3193 | 92.0 | -0.6807 |

| shot | Pixel AP | Image AP |
|---:|---:|---:|
| 2 | 35.4267 | 89.5327 |
| 16 | 38.6827 | 95.5620 |
| 64 | 41.5340 | 96.1640 |

随着 shot 数增加，本次 pixel AUROC、pixel AP、image AUROC 和 image AP 总体均上升。2-shot image AUROC 的明显偏差在 16-shot 和 64-shot 中消失，进一步支持“极低样本结果对未公开抽样索引非常敏感”的判断。

### 11.8 VisA full-shot → MVTec

该实验使用官方 VisA full-shot 清单的 2,162 条记录。清单 SHA-256 为 f8eaf8224a20e80ec8301ca8cd0f17b8e3c53ebe1cb45f643ad38198946f3efe，训练目录为 runs/VisA_full_shot_0shot_seed111/。

| 阶段 | 开始 | 完成 | 耗时 | 初始 loss | 最终 loss |
|---|---|---|---:|---:|---:|
| 文本 + 图像训练 | 2026-08-15 17:50:05 | 2026-08-15 21:42:06 | 3 小时 52 分 01 秒 | 文本 1.1487；图像 3.1378 | 文本 0.9631；图像 2.2699 |
| MVTec epoch 15 测试 | 2026-08-15 21:42:57 | 2026-08-15 21:52:14 | 约 9 分 17 秒 | — | — |

图像阶段 20 个 epoch 的 loss 为：

    3.13776, 2.82844, 2.75315, 2.70858, 2.65635,
    2.60429, 2.57748, 2.56142, 2.51221, 2.48979,
    2.44101, 2.42251, 2.40345, 2.37205, 2.37557,
    2.33473, 2.29800, 2.28742, 2.27646, 2.26989

按作者对 full-shot checkpoint 的说明，预先固定使用 epoch 15 在 MVTec 上测试，没有事后挑选最好 checkpoint。逐类结果如下：

| 类别 | Pixel AUROC | Pixel AP | Image AUROC | Image AP |
|---|---:|---:|---:|---:|
| bottle | 93.78 | 62.69 | 93.97 | 98.30 |
| cable | 84.98 | 27.73 | 80.25 | 88.78 |
| capsule | 96.03 | 29.49 | 81.89 | 95.79 |
| carpet | 99.57 | 77.71 | 99.44 | 99.84 |
| grid | 97.68 | 33.56 | 99.25 | 99.71 |
| hazelnut | 97.64 | 62.49 | 85.32 | 92.39 |
| leather | 99.37 | 53.32 | 100.00 | 100.00 |
| metal_nut | 70.55 | 24.48 | 78.20 | 94.90 |
| pill | 86.62 | 25.71 | 81.89 | 96.06 |
| screw | 98.59 | 39.68 | 91.64 | 97.18 |
| tile | 93.35 | 71.17 | 96.72 | 98.88 |
| transistor | 73.05 | 13.53 | 71.21 | 68.46 |
| toothbrush | 96.09 | 44.50 | 94.44 | 97.96 |
| wood | 98.04 | 69.04 | 98.77 | 99.64 |
| zipper | 96.55 | 48.07 | 88.60 | 96.96 |
| **平均** | **92.1260** | **45.5447** | **89.4393** | **94.9900** |

| 指标 | 本次 | 论文 full-shot | 差值 |
|---|---:|---:|---:|
| Pixel AUROC | 92.1260 | 91.9 | +0.2260 |
| Image AUROC | 89.4393 | 90.5 | -1.0607 |

pixel AUROC 略高于论文，image AUROC 相差约 1.06 个百分点；在未公开 full-shot 训练细节和完整 checkpoint 选择规则的前提下，可以视为高度一致的近似复现。

### 11.9 MVTec 2/16/64/full-shot → VisA

反向协议使用 MVTec 训练、VisA 测试。2/16/64-shot 均使用 seed 111；few-shot 清单生成细节和 toothbrush 重复取样限制见第 10 节。full-shot 使用官方 MVTec full-shot 清单，共 1,725 条记录，SHA-256 为 374015dbecfe469af89a51d36e1d99a8e9771d20975af9332e128d581994bb9f。

#### 11.9.1 训练时间与收敛

| 设置 | 训练开始 | 训练完成 | 耗时 | 文本 loss | 图像 loss | 测试 epoch |
|---|---|---|---:|---:|---:|---:|
| 2-shot | 2026-08-15 21:52:15 | 21:56:27 | 4 分 12 秒 | 1.3994 → 1.2187 | 5.3667 → 2.2206 | 20 |
| 16-shot | 2026-08-15 22:07:58 | 22:34:50 | 26 分 52 秒 | 1.2634 → 1.1617 | 4.6680 → 2.1097 | 20 |
| 64-shot | 2026-08-15 22:46:25 | 2026-08-16 00:31:13 | 1 小时 44 分 48 秒 | 1.1979 → 1.1536 | 4.0090 → 1.9236 | 20 |
| full-shot | 2026-08-16 00:42:57 | 03:49:28 | 3 小时 06 分 31 秒 | 1.1993 → 1.1472 | 3.6021 → 1.9908 | 15 |

各训练均完成 5 个文本 epoch、20 个图像 epoch，并生成 1 个文本适配器和 20 个图像适配器 checkpoint；新增运行日志中没有 OOM、CUDA、NCCL、RuntimeError 或数据读取错误。

MVTec full-shot 图像阶段 20 个 epoch 的 loss 为：

    3.60211, 2.99650, 2.75236, 2.62696, 2.53563,
    2.43874, 2.38499, 2.35894, 2.29267, 2.24653,
    2.22699, 2.21013, 2.17461, 2.12744, 2.13308,
    2.08766, 2.07512, 2.08107, 2.02445, 1.99085

#### 11.9.2 2-shot 在 VisA 上的逐类结果

| 类别 | Pixel AUROC | Pixel AP | Image AUROC | Image AP |
|---|---:|---:|---:|---:|
| candle | 98.54 | 21.65 | 76.93 | 79.98 |
| pcb3 | 91.20 | 11.74 | 62.22 | 64.58 |
| capsules | 94.97 | 23.65 | 59.75 | 75.80 |
| pipe_fryum | 96.14 | 18.83 | 80.76 | 90.12 |
| pcb4 | 94.69 | 34.03 | 86.97 | 89.37 |
| macaroni2 | 97.84 | 6.10 | 66.29 | 66.42 |
| pcb2 | 91.10 | 14.51 | 70.27 | 72.92 |
| chewinggum | 99.55 | 76.15 | 97.20 | 98.76 |
| macaroni1 | 97.76 | 12.00 | 75.20 | 78.05 |
| cashew | 95.61 | 29.33 | 80.16 | 90.35 |
| fryum | 93.97 | 22.72 | 79.48 | 89.93 |
| pcb1 | 83.08 | 3.64 | 87.10 | 84.96 |
| **平均** | **94.5375** | **22.8625** | **76.8608** | **81.7700** |

#### 11.9.3 16-shot 在 VisA 上的逐类结果

| 类别 | Pixel AUROC | Pixel AP | Image AUROC | Image AP |
|---|---:|---:|---:|---:|
| candle | 97.38 | 32.15 | 87.52 | 89.88 |
| pcb3 | 91.10 | 13.48 | 65.24 | 71.58 |
| capsules | 96.02 | 32.56 | 66.35 | 80.60 |
| pipe_fryum | 96.36 | 25.37 | 93.18 | 96.47 |
| pcb4 | 94.04 | 19.30 | 87.63 | 89.75 |
| macaroni2 | 96.92 | 1.81 | 69.43 | 70.02 |
| pcb2 | 91.39 | 10.53 | 72.79 | 75.45 |
| chewinggum | 99.58 | 79.01 | 97.64 | 98.99 |
| macaroni1 | 97.12 | 10.79 | 79.22 | 79.38 |
| cashew | 93.11 | 22.18 | 87.00 | 93.77 |
| fryum | 92.74 | 19.84 | 76.84 | 89.58 |
| pcb1 | 88.35 | 3.93 | 82.54 | 85.92 |
| **平均** | **94.5092** | **22.5792** | **80.4483** | **85.1158** |

#### 11.9.4 64-shot 在 VisA 上的逐类结果

| 类别 | Pixel AUROC | Pixel AP | Image AUROC | Image AP |
|---|---:|---:|---:|---:|
| candle | 97.33 | 31.65 | 76.50 | 82.34 |
| pcb3 | 91.12 | 12.68 | 73.42 | 77.84 |
| capsules | 95.28 | 30.34 | 73.03 | 84.40 |
| pipe_fryum | 97.26 | 32.38 | 93.66 | 96.53 |
| pcb4 | 94.42 | 19.82 | 92.34 | 91.32 |
| macaroni2 | 96.39 | 1.01 | 65.02 | 67.20 |
| pcb2 | 89.92 | 8.10 | 76.31 | 78.52 |
| chewinggum | 99.36 | 81.20 | 95.84 | 98.33 |
| macaroni1 | 97.33 | 8.56 | 85.14 | 84.43 |
| cashew | 94.42 | 30.04 | 86.70 | 94.03 |
| fryum | 93.99 | 25.52 | 83.30 | 92.02 |
| pcb1 | 90.07 | 3.98 | 81.04 | 81.88 |
| **平均** | **94.7408** | **23.7733** | **81.8583** | **85.7367** |

#### 11.9.5 full-shot 在 VisA 上的逐类结果

| 类别 | Pixel AUROC | Pixel AP | Image AUROC | Image AP |
|---|---:|---:|---:|---:|
| candle | 97.87 | 38.22 | 74.41 | 80.77 |
| pcb3 | 90.36 | 14.39 | 68.91 | 69.73 |
| capsules | 95.69 | 32.44 | 67.10 | 81.27 |
| pipe_fryum | 97.76 | 36.01 | 88.42 | 93.26 |
| pcb4 | 93.94 | 16.45 | 91.59 | 90.91 |
| macaroni2 | 97.19 | 0.96 | 67.40 | 66.64 |
| pcb2 | 91.09 | 6.96 | 68.92 | 68.18 |
| chewinggum | 99.46 | 82.81 | 94.06 | 97.55 |
| macaroni1 | 97.27 | 7.94 | 81.17 | 80.58 |
| cashew | 93.88 | 23.32 | 88.02 | 94.33 |
| fryum | 94.02 | 28.07 | 85.20 | 92.52 |
| pcb1 | 89.89 | 4.22 | 46.42 | 51.21 |
| **平均** | **94.8683** | **24.3158** | **76.8017** | **80.5792** |

#### 11.9.6 与论文 VisA 结果比较

论文表 1/表 2 说明 VisA 指标来自 MVTec 训练模型；其 2/16/64/full-shot 的 pixel AUROC 为 93.4/93.8/94.0/95.5，image AUROC 为 78.4/84.0/84.1/84.6。

| shot | 本次 Pixel AUROC | 论文 | 差值 | 本次 Image AUROC | 论文 | 差值 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 94.5375 | 93.4 | +1.1375 | 76.8608 | 78.4 | -1.5392 |
| 16 | 94.5092 | 93.8 | +0.7092 | 80.4483 | 84.0 | -3.5517 |
| 64 | 94.7408 | 94.0 | +0.7408 | 81.8583 | 84.1 | -2.2417 |
| full | 94.8683 | 95.5 | -0.6317 | 76.8017 | 84.6 | -7.7983 |

定位能力（pixel AUROC）四档均与论文相差不超过 1.14 个百分点，其中前三档还略高于论文。图像级结果则明显更敏感：2/16/64-shot 随样本数增加而改善，但 full-shot 回落。逐类表显示 full-shot 的 pcb1 image AUROC 仅为 46.42，是平均值回落的重要来源。由于作者没有发布 few-shot 文件索引，也未完整公开 full-shot 的类别采样/平衡和 checkpoint 细节，本结果证明代码和双向跨数据集链路可运行，但不足以宣称逐位还原论文 VisA 表格。

### 11.10 全部 results.csv 汇总与流水线审计（ai写了流水线帮忙顶着）

以下逐行汇总远端 8 个 results.csv 的全部记录；VisA 2-shot 文件包含一次同数据自测以及两个 MVTec checkpoint 结果，其余文件各包含一条正式结果。

| 训练运行 | 测试集 | epoch | Pixel AUROC | Pixel AP | Image AUROC | Image AP |
|---|---|---:|---:|---:|---:|---:|
| VisA 2-shot | VisA（工程自测） | 20 | 93.4317 | 26.5908 | 80.7975 | 84.5867 |
| VisA 2-shot | MVTec | 20 | 90.0167 | 35.4267 | 78.2680 | 89.5327 |
| VisA 2-shot | MVTec | 15 | 89.8280 | 35.4473 | 77.3413 | 89.2100 |
| VisA 16-shot | MVTec | 20 | 90.8167 | 38.6827 | 90.3640 | 95.5620 |
| VisA 64-shot | MVTec | 20 | 91.3373 | 41.5340 | 91.3193 | 96.1640 |
| VisA full-shot | MVTec | 15 | 92.1260 | 45.5447 | 89.4393 | 94.9900 |
| MVTec 2-shot | VisA | 20 | 94.5375 | 22.8625 | 76.8608 | 81.7700 |
| MVTec 16-shot | VisA | 20 | 94.5092 | 22.5792 | 80.4483 | 85.1158 |
| MVTec 64-shot | VisA | 20 | 94.7408 | 23.7733 | 81.8583 | 85.7367 |
| MVTec full-shot | VisA | 15 | 94.8683 | 24.3158 | 76.8017 | 80.5792 |

本轮总流水线从 2026-08-15 17:50:05 的 VisA full-shot 训练开始，到 2026-08-16 04:01:06 写入 stage=pipeline_completed，历时约 10 小时 11 分。结束时 screen 会话为 0、GPU 利用率 0%、显存占用 1 MiB；数据盘剩余约 14 GB。

关键审计哈希：

| 文件 | SHA-256 |
|---|---|
| scripts/run_remaining_core.sh | 0f779424e857efe1e75dade2aa9978c380b3984e0c24175cfcfecaf17ea46513 |
| runs/remaining-core-pipeline.log | ce22d457b9281aa9edbe9a64725be73fa215210b50b9ce3c238c67827c871b12 |
| VisA_few_shot_2shot_seed111/results.csv | 3d3193a5dea25c668c18a6a7b71b8acab6f84500cb499bbd74e93338072a8043 |
| VisA_few_shot_16shot_seed111/results.csv | 5ffd90a0f817428c5af100a7e56ede66580c9b28bc44023b58b552b8a76ddb0a |
| VisA_few_shot_64shot_seed111/results.csv | 8a2aef839bfd8801fe17e6b4709a671935b523eb6fa16d7cc39ce2ee54dc987e |
| VisA_full_shot_0shot_seed111/results.csv | 0ed818ec271efe95c89c48bec9bd439606c2e921c4ab4d5a92248e2a63aadcf7 |
| MVTec_few_shot_2shot_seed111/results.csv | 6c0fe5ffb36862025170305e17f531dcf2c2b66c32be8164010bedb6ce23f2a6 |
| MVTec_few_shot_16shot_seed111/results.csv | 7056bbd75efb27838e07d80e379d07199d055ab45218cc5f898473fa6bc5639a |
| MVTec_few_shot_64shot_seed111/results.csv | d8fbbc79650918460096a4308ee7c8e84150b8f855703b4fb2d7ce8eaf62c644 |
| MVTec_full_shot_0shot_seed111/results.csv | 00de03ee04194e72206e7e6be896a55ae847e9612fa85cec643f810ceabbb2de |

最终核验结果：新增 5 个运行目录中的错误关键字匹配数为 0，总流水线日志中的错误关键字匹配数也为 0。

## 12. 论文描述与官方代码的差异

本次没有擅自改变官方损失逻辑。

### 12.1 第一阶段缺少论文所写的分类损失

论文公式把分类损失和分割损失都写入第一阶段的对齐损失，但当前官方 `train_text_adapter` 没有计算分类损失。仓库 README 的追加讨论表示作者后来选择依靠分割监督，GitHub Issue #38 中也有人报告加入分类损失会改变 pixel/image AUROC 的取舍。

本次选择：保持当前官方代码行为，不自行加入分类损失。

### 12.2 文本阶段四层 patch loss 实际只反传最后一层

`train_text_adapter` 在遍历四层 patch 特征时重复赋值 `loss`，而不是累加：

```python
for f in patch_features:
    loss = calculate_seg_loss(...)
```

因此当前代码最终只对最后一次计算的 loss 调用 `backward()`。这与论文中多层特征组合的文字表述不完全一致，相关问题在官方 Issues 中仍有讨论。

本次选择：为了复现“当前发布代码”，不改变这一逻辑；同时在报告中明确记录。

### 12.3 checkpoint 选择不透明

官方测试脚本遍历 20 个 checkpoint。作者在 Issue #13 中确认 full-shot 论文结果来自 epoch 15，但没有公开 2/16/64-shot 的统一选择规则；Issue #52 对“是否针对每个数据集挑最好 epoch”的问题仍无作者答复。

本次选择：明确报告所用 epoch，不在 20 个 checkpoint 中按测试集指标进行隐式挑选。

### 12.4 few-shot 样本索引未公开

作者只说明每类随机选取等量正常和异常样本，没有公布具体文件列表。2-shot 对单个样本非常敏感，因此即使环境、代码和 seed 一致，也无法保证逐位复原论文数字。

本次选择：发布清单哈希，确保本次实验本身可重复、可审计。

## 13. 自动化脚本与常用命令

远端脚本位于：

```text
/root/autodl-tmp/AA-CLIP/scripts/
```

主要脚本：

| 脚本 | 用途 |
|---|---|
| `setup_env.sh` | 创建并验证固定 Python/CUDA 环境 |
| `download_clip_weight.sh` | 断点下载并校验 CLIP 权重 |
| `smoke_model.py` | 518×518 模型前向冒烟测试 |
| `verify_env.py` | 检查 PyTorch、CUDA、cuDNN 和依赖 |
| `download_visa.sh` | 并行下载、校验并解压 VisA |
| `install_mvtec_archive.sh` | 校验用户取得的 MVTec tar 并安装 |
| `verify_dataset.py` | 逐条检查 metadata 引用文件 |
| `generate_few_shot.py` | 生成确定性 few-shot 清单 |
| `run_train.sh` | 保存日志和 checkpoint 的可恢复训练 |
| `run_test.sh` | 测试指定数据集和 checkpoint |
| `summarize_results.py` | 从 `test.log` 提取 CSV |
| `run_remaining_core.sh` | 顺序衔接 VisA full-shot 与 MVTec 2/16/64/full-shot 的训练、测试和汇总；失败即停止 |

用到的命令：

```bash
cd /root/autodl-tmp/AA-CLIP

# 环境与模型检查
/root/autodl-tmp/envs/aaclip/bin/python scripts/verify_env.py
/root/autodl-tmp/envs/aaclip/bin/python scripts/smoke_model.py

# 数据检查
/root/autodl-tmp/envs/aaclip/bin/python scripts/verify_dataset.py \
  --dataset VisA --manifest 16-shot.jsonl
/root/autodl-tmp/envs/aaclip/bin/python scripts/verify_dataset.py \
  --dataset MVTec --manifest full-shot.jsonl

# 训练
./scripts/run_train.sh VisA 16 111 few_shot
./scripts/run_train.sh VisA 64 111 few_shot

# MVTec 测试 epoch 20
./scripts/run_test.sh runs/VisA_few_shot_16shot_seed111 MVTec 111 20
./scripts/run_test.sh runs/VisA_few_shot_64shot_seed111 MVTec 111 20

# 提取结果
/root/autodl-tmp/envs/aaclip/bin/python scripts/summarize_results.py \
  runs/VisA_few_shot_16shot_seed111/test.log \
  --output runs/VisA_few_shot_16shot_seed111/results.csv

# 本轮完整顺序流水线（运行前会拒绝覆盖已有目录）
./scripts/run_remaining_core.sh
```

查看后台任务：

```bash
screen -ls
tail -f runs/VisA_few_shot_16shot_seed111/console.log
tail -f runs/VisA_few_shot_16shot_seed111/train.log
tail -f runs/remaining-core-pipeline.log
```

## 14. 文件和产物位置

远端项目：

```text
/root/autodl-tmp/AA-CLIP
```

数据：

```text
/root/autodl-tmp/AA-CLIP/data/VisA_20220922
/root/autodl-tmp/AA-CLIP/data/mvtec_ad
```

VisA 2-shot 产物：

```text
/root/autodl-tmp/AA-CLIP/runs/VisA_few_shot_2shot_seed111/
├── text_adapter.pth
├── image_adapter.pth
├── image_adapter_1.pth ... image_adapter_20.pth
├── train.log
├── console.log
├── test.log
├── test-console.log
├── selftest-results.csv
└── results.csv
```

VisA 16-shot 产物：

```text
/root/autodl-tmp/AA-CLIP/runs/VisA_few_shot_16shot_seed111/
├── text_adapter.pth
├── image_adapter.pth
├── image_adapter_1.pth ... image_adapter_20.pth
├── train.log
├── console.log
├── test.log
├── test-console.log
└── results.csv
```

VisA 64-shot 产物：

```text
/root/autodl-tmp/AA-CLIP/runs/VisA_few_shot_64shot_seed111/
├── text_adapter.pth
├── image_adapter.pth
├── image_adapter_1.pth ... image_adapter_20.pth
├── train.log
├── console.log
├── test.log
├── test-console.log
└── results.csv
```

VisA full-shot 与反向协议产物：

```text
/root/autodl-tmp/AA-CLIP/runs/VisA_full_shot_0shot_seed111/
/root/autodl-tmp/AA-CLIP/runs/MVTec_few_shot_2shot_seed111/
/root/autodl-tmp/AA-CLIP/runs/MVTec_few_shot_16shot_seed111/
/root/autodl-tmp/AA-CLIP/runs/MVTec_few_shot_64shot_seed111/
/root/autodl-tmp/AA-CLIP/runs/MVTec_full_shot_0shot_seed111/
```

上述每个目录均包含 text_adapter.pth、image_adapter.pth、image_adapter_1.pth 至 image_adapter_20.pth、train.log、console.log、test.log、test-console.log 和 results.csv。

总流水线与阶段日志：

```text
/root/autodl-tmp/AA-CLIP/scripts/run_remaining_core.sh
/root/autodl-tmp/AA-CLIP/runs/remaining-core-pipeline.log
```

## 15. 复现评价

本次已经完成论文最核心的两条工业跨数据集链路：VisA 2/16/64/full-shot → MVTec，以及 MVTec 2/16/64/full-shot → VisA。环境、模型权重、数据、两阶段训练、checkpoint、推理和指标汇总均在论文同款单张 RTX 3090 24 GB 上真实执行，并保留了可恢复日志和哈希。

VisA → MVTec 的 16-shot、64-shot 和 full-shot 与论文高度接近；full-shot pixel/image AUROC 的差值分别为 +0.2260/-1.0607 个百分点。反向 MVTec → VisA 的四档 pixel AUROC 与论文均相差不超过 1.14 个百分点，说明异常定位链路稳定复现；image AUROC 的差异更明显，尤其 full-shot 低 7.7983 个百分点，表明全局异常分数对样本组成、类别平衡和 checkpoint 更敏感。

因此，最严谨的结论是：

> 已复现 AA-CLIP 官方代码在 VisA 与 MVTec 间的双向 2/16/64/full-shot 训练和 zero-shot 评估主链路，并得到可审计、像素级高度一致的近似结果；由于作者未公开实际 few-shot 索引及全部采样和选点细节，不能声称逐位复原论文全部数字。

尚未运行的是 BTAD、MPDD 和论文中的医学数据集，因为这些数据尚未准备；

进一步提高可信度的合理顺序是：

1. 对 2-shot 使用多个预先声明的随机 seed，报告均值与标准差；
2. 对 MVTec full-shot 构造与 few-shot 一致的 1:1 类别平衡版本，并与官方 full-shot 清单分开报告；
3. 在不根据测试集挑 checkpoint 的前提下，预注册 epoch 15/20 的对照实验；
4. 获取 BTAD、MPDD 等数据后检查跨数据集趋势；
5. 将“原样官方 loss”和“按论文公式修正 loss”作为两个明确分开的实验分支。

## 16. 参考链接

- 论文：https://arxiv.org/abs/2503.06661
- 官方代码：https://github.com/Mwxinnn/AA-CLIP
- VisA：https://github.com/amazon-science/spot-diff
- MVTec AD：https://www.mvtec.com/research-teaching/datasets/mvtec-ad
- full-shot epoch 15 说明：https://github.com/Mwxinnn/AA-CLIP/issues/13
- 第一阶段 loss 讨论：https://github.com/Mwxinnn/AA-CLIP/issues/38
- few-shot 定义讨论：https://github.com/Mwxinnn/AA-CLIP/issues/8
- checkpoint 选择疑问：https://github.com/Mwxinnn/AA-CLIP/issues/52
