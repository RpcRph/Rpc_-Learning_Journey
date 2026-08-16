# AA-CLIP 论文转述与汇报

本目录记录对 CVPR 2025 论文 *AA-CLIP: Enhancing Zero-Shot Anomaly Detection via Anomaly-Aware CLIP* 的双语转述与演示材料。中英文文字稿分别提供 Word 和 LaTeX 两种正规学术论文版本，均以自己的语言聚焦研究问题、核心方法、关键结果、批判性讨论与个人研究判断；LaTeX 版各 2 页，Word 版各 3 页。

## 作业内容

- 中文迷你论文：[Word 文档](AA-CLIP_论文转述_中文版.docx) · [LaTeX 源文件](AA-CLIP_论文转述_中文版.tex) · [PDF 成品](AA-CLIP_论文转述_中文版.pdf)
- English mini-paper: [Word document](AA-CLIP_Paper_Paraphrase_English.docx) · [LaTeX source](AA-CLIP_Paper_Paraphrase_English.tex) · [PDF](AA-CLIP_Paper_Paraphrase_English.pdf)
- [双语演示文稿](AA-CLIP_论文总结汇报_Rpc.pptx)
- [论文图片来源说明](assets/README.md)

四份文字稿均包含题目、作者、摘要、关键词、数学公式、方法图、实验表格和规范参考文献，正文统一采用“引言—方法—实验与结果—讨论—结论”的五段式结构。讨论部分进一步加入我对工业异常检测、高效适配和跨域可靠性的理解，不把尚未完成的复现写成已有成果。LaTeX 使用 XeLaTeX 编译；中间文件建议统一输出到 `build/`，避免污染作业目录。

```bash
mkdir -p build
latexmk -xelatex -outdir=build 'AA-CLIP_论文转述_中文版.tex'
latexmk -xelatex -outdir=build 'AA-CLIP_Paper_Paraphrase_English.tex'
```

## 演示文稿结构

演示文稿共 11 页，以中文为主、英文关键词为辅：

1. 论文与汇报主题
2. 研究问题：CLIP 的“异常无感知”
3. AA-CLIP 的总体思路
4. 阶段一：解耦正常/异常文本锚点
5. 阶段二：让图像块特征对齐文本锚点
6. 实验设计与主要结果
7. 定性结果与消融证据
8. 我的评价：价值、边界与可延伸问题
9. 我的当前研究现况：方向选择、现有基础与近期计划
10. 我的理解：研究问题、工作假设与验证方法
11. 总结

第 9 页明确区分已经完成的学习工作与尚未开展的领域复现；第 10 页不是继续罗列文献，而是提出“语义锚点—受控视觉残差”的个人工作假设，并给出跨数据集、消融、校准误差、显存、延迟和失败样例等验证维度。两页后续都可以随着实际研究进展继续更新。

## 原始资料

- 论文页面：<https://arxiv.org/abs/2503.06661>
- 官方代码：<https://github.com/Mwxinnn/AA-CLIP>
- 发表信息：CVPR 2025

本文字稿属于学习性转述，不代表原论文作者观点之外的新实验结果；所有实验数字与论文图片均来自原论文。
