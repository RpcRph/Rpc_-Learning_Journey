# 课程学习报告

- 完成日期：2026-07-18
- 课程：学业基本情况介绍与计算机基本使用
- 状态：已完成

## 文件

- [课程学习报告_任鹏程.tex](课程学习报告_任鹏程.tex)：LaTeX 源文件。
- [课程学习报告_任鹏程.pdf](课程学习报告_任鹏程.pdf)：提交版 PDF。

## 内容概览

报告总结了研究生阶段的学习认识，并整理了 Linux、Markdown、LaTeX、Java、JavaScript、Python 和 C++ 等基础知识与使用经验。

## 编译方式

文档使用 `ctexart` 和 Windows 中文字体配置，应使用 XeLaTeX 编译。目录与总页数需要至少两次编译才能正确更新：

```powershell
xelatex .\课程学习报告_任鹏程.tex
xelatex .\课程学习报告_任鹏程.tex
```

编译生成的临时文件已由仓库根目录的 `.gitignore` 排除，PDF 成品仍会被版本控制。

