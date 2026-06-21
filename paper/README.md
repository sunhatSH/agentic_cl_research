# paper/

Everything for the paper "Continual Reinforcement Learning of Multi-Turn
Agentic LLMs" lives here, separated from the design docs in `../doc/`.

```
paper/
├── drafts/     Markdown prose drafts (the writing source of truth)
│   ├── Paper_论文向总览.md         总览/索引：摘要·贡献·方法·实验·系统·局限 + 文档地图
│   ├── Paper_Intro_draft_{CN,EN}.md   §1 Introduction（单一论点：先净化信号、再巩固）
│   └── Paper_Method_draft_{CN,EN}.md  §4 Method（5 子节，两幕结构 + 附录 A 三 prompt）
├── latex/      Submission LaTeX (assembled from drafts)
│   ├── main.tex
│   └── sections/  01_introduction · 04_method · 05_experiments · A_prompts
├── assets/     Figures (Fig 1–5; see assets/README.md)
└── refs/        references.bib (keys A1–A10 / B1–B6 / C1–C3 / D1 match the [A2]/[B4] tags)
```

## 信源约定

- **散文以 `drafts/Paper_*_EN.md` 为准**（投稿用英文；CN 为对照）。`latex/sections/*`
  是骨架 + `\TODO`，从 drafts 移植，不重复维护整段散文（避免双份漂移）。
- **公式/超参/文献以 `../doc/CL_Update_Sunhao.md` 为单一信源**；drafts 通过
  `../../doc/...` 相对链接引用设计文档，不复制其内容。
- **三个 agent 的 prompt 以 `../agents/prompts.py` 为准**（已实现）；附录 A 逐字粘贴。
- 依赖实验结果的数字一律 `[TODO: 训练后回填]` / `\TODO{...}`。

## 构建

```bash
cd paper/latex && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

> 当前 `latex/` 是骨架（section 体为 `\TODO` + 结构注释）；正式入稿时从 drafts 移植散文、补 Fig 1–5、回填 Results。
