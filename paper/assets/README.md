# Figures (assets/)

Figure source files for the paper. `main.tex` has `\graphicspath{{../assets/}}`,
so a `\includegraphics{fig1_overview}` resolves here. Keep editable sources
(`.drawio` / `.svg` / `.py`) next to the exported `.pdf`/`.png`.

Figure list (captions are the source of truth in `../drafts/Paper_Method_draft_EN.md`):

| File (suggested) | Figure | Content |
|------------------|--------|---------|
| `fig1_overview.pdf`   | Fig 1 | System overview: clean-signal rollout (16×8 + winner-sync + 3-agent loop) → CL trainer (L_cl) with 7-bucket buffer; dashed "clean advantage" arrow links the two halves. |
| `fig2_buckets.pdf`    | Fig 2 | 7 capability buckets + task counts + square-root quota at 25k capacity. |
| `fig3_ushape.pdf`     | Fig 3 | U-shaped block weight vs normalized block position, γ=δ∈{1.0, 0.92, 0.88}. |
| `fig4_winner_sync.pdf`| Fig 4 | Winner-synchronized session timeline (spawn 8 → rollout → winner → sync → next query → destroy). |
| `fig5_three_agent.pdf`| Fig 5 | 3-agent loop at a winner-sync boundary: observer → report R_t → {reward, questioner}. |

All five are `[TODO: draw before submission]`.
