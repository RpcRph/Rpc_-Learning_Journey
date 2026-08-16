### 当前版本

- ROCm：7.14.0
- PyTorch：2.12.0+rocm7.14.0
- torchvision：0.27.0+rocm7.14.0
- torchaudio：2.11.0+rocm7.14.0
- Triton：3.7.1 ROCm 7.14
- bitsandbytes：0.50.1
- transformers：5.15.0
- accelerate：1.14.0
- peft：0.20.0
- datasets：5.0.1
- trl：1.10.0

这是 AMD 当前 ROCm 7.14 对应的稳定 PyTorch 组合，而旧的 PyTorch 2.13+ROCm 7.2 更接近开发版本组合。[ROCm 7.14 发布说明](https://rocm.docs.amd.com/en/docs-7.14.0/about/release-notes.html)、[AMD PyTorch 安装说明](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)。

系统 `/opt/rocm` 是 Radeon 26.13 仓库提供的 `7.14.0~pre3`，已经是该仓库当前候选版本；训练时实际加载的是 Conda 环境内精确的 ROCm 7.14.0 库，因此 PyTorch 和运行库版本已对应。

### RX 6650 XT 映射

- 原生 `gfx1032`：能识别 RX 6650 XT，但第一次实际 GPU 计算会卡住。
- 映射为 `gfx1030`：矩阵、卷积、反向传播和优化器更新全部正常。

保留 `HSA_OVERRIDE_GFX_VERSION=10.3.0`，并把所需兼容参数写进 `train` 环境，激活后自动生效。

已通过：

- FP32 矩阵计算与反向传播
- FP16 线性层、反向传播、AdamW 更新
- MIOpen 卷积反向传播
- bitsandbytes NF4 4-bit
- PyTorch 依赖完整性检查

4-bit 可以用于 QLoRA。8-bit `Linear8bitLt` 仍因缺少 `gfx1030` 的 hipBLASLt 内核而失败；bitsandbytes 的官方 ROCm 支持目前仍是预览状态，预编译目标也没有包含 gfx1030，因此暂时不使用 `load_in_8bit=True`。[bitsandbytes ROCm 说明](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/docs/source/installation.mdx)

### 目前的一些警告

- `Windows driver is old`：误判。实际驱动 `32.0.21045.1000` 正是当前 Adrenalin 26.7.1 对应版本。[AMD 26.7.1 说明](https://www.amd.com/de/resources/support-articles/release-notes/RN-RAD-WIN-26-7-1.html)
- `No WDDM adapters found`：当前 ROCDXG/非官方显卡兼容路径下的非致命提示，GPU 计算已经实测通过。
- MIOpen 数据库缺失提示：会自动回退搜索算法，可能让首次卷积启动稍慢，不影响正确性。
- `rocSHMEM Could not open libnuma`：不影响单卡训练。

现在直接使用：

```bash
wsl
conda activate train
```

重新检查环境：

```bash
python /mnt/c/Users/Rpc/Documents/Codex/2026-08-14/wo-xi/rocm_training_smoke_test.py
```

测试脚本位于 [rocm_training_smoke_test.py](C:/Users/Rpc/Documents/Codex/2026-08-14/wo-xi/rocm_training_smoke_test.py)。

旧环境没有删除，保留为 `train-rocm72-old`，另外还有 `train-backup-20260814` 备份。 8GB 显存建议优先采用 FP16 LoRA 或 NF4 4-bit QLoRA，并开启 gradient checkpointing。