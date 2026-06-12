# Paper Draft — Thesis & Introduction (§1), English

> **Use**: the single-thesis framing + Introduction prose for the *whole project as one paper*. The point of this draft is discipline: **one core message, with all five components demoted to "how we realize it"**, so the paper reads as one coherent contribution rather than a feature list.
> **Conventions**: results-dependent claims marked `[TODO: results]`; citation tags `[A2]`/`[B4]`/`[B2]`/`[C1]` match [`CL_Update_Sunhao.md`](../../doc/CL_Update_Sunhao.md). Chinese version: [`Paper_Intro_draft_CN.md`](Paper_Intro_draft_CN.md).
> **Date**: 2026-06-12

---

## The one-sentence thesis (memorize; everything serves this)

> **In continual RL of multi-turn agentic LLMs, preventing catastrophic forgetting requires not only consolidating the learning signal, but first guaranteeing that the signal is uncorrupted at its source.**

Everything in the paper is one of two things: (a) *making the signal clean at the source*, or (b) *consolidating that clean signal without forgetting*. No component is presented as a standalone trick.

---

## 1 Introduction

Deployed LLM agents improve through periodic reinforcement-learning updates: every few weeks, freshly collected interaction data is used to refine the policy. This is, in effect, *continual reinforcement learning*—the model must acquire new behaviors while retaining the capabilities it already has. Two failure modes make this hard in the agentic, multi-turn setting. The first is **catastrophic forgetting**: updates on new tasks erode performance on previously mastered domains. The second is **Echo Trap** [B4]: in multi-turn agentic RL, policy diversity collapses, within-group rollouts become near-identical, and training stalls or diverges. Reinforcement fine-tuning is known to forget less than supervised fine-tuning [B2, C1], but it still forgets, and the multi-turn agentic regime introduces failure modes that single-turn studies do not surface.

The dominant remedy for forgetting—experience replay [A2]—and its refinements (prioritized [A9], partitioned [A3], compressed [A8] buffers) all share an implicit assumption: that the replayed and on-policy experiences carry a *faithful* learning signal. We argue that in continual **multi-turn agentic** RL this assumption silently breaks, and that the break happens *before* any replay or regularization is applied. Group-relative policy optimization (GRPO), the workhorse for these agents, estimates each sample's advantage by normalizing rewards within a group of $M$ rollouts of the same query. This estimator is unbiased *only* when the $M$ trajectories differ purely because of stochastic policy sampling. Two characteristically agentic effects violate this precondition at the source:

1. **Within-group environment divergence.** Across a multi-query session, the $M$ execution environments drift apart; a slot can be penalized not for a worse policy but for an unfavorable state left by an earlier query, so the advantage absorbs environment noise that accumulates with turn index.
2. **Premise drift in multi-turn data.** A follow-up query typically references the previous turn's result ("the figures on page 3 are wrong"). When the follow-up is fixed at data-collection time but the realized state is a stochastic, policy-dependent function, its premise holds only with a probability that *drifts* as the policy improves—rewarding hallucinated fixes or punishing honest "there is no such problem" answers.

A corrupted advantage cannot be repaired downstream: no replay weighting or KL anchor recovers a gradient whose sign was wrong to begin with. This reframes continual agentic RL as a **two-stage** problem—first produce a clean signal, then consolidate it—and organizes our entire system around that thesis.

**Making the signal clean at the source.** We enforce the GRPO precondition with a sandboxed rollout system in which every group of $M$ slots is derived from a single master so they start bit-identical, and the session is re-aligned to the *winning* trajectory at each turn boundary, so the next query's group again starts from a common state (eliminating divergence #1). We eliminate premise drift (#2) by retaining only the first query of each real session as a seed—the real follow-ups in the data are themselves bound to the original execution and would suffer the same drift if reused—and constructing every subsequent turn **online, after observing the realized state**, with three cooperating agents: a persona-free **observer** that gathers an objective report of the realized intermediate and final results, a persona-conditioned **questioner** that poses the next query from that report, and a **reward model** that scores the turn from the same report. Because each follow-up is written after the state is observed, its premise is grounded by construction.

**Consolidating the clean signal without forgetting.** On top of the cleaned signal we apply a four-term continual-learning objective (RL, reverse-KL, replay, entropy) injected into GRPO without forking the trainer; a replay buffer partitioned by *capability* rather than difficulty and prioritized by *anti-forgetting value* rather than reward; and a U-shaped, action-block-level reweighting of the replay loss that concentrates credit on the early branching decisions and the concluding segment of each trajectory.

We validate the full system on a 27B agent across seven capability domains, using a Pass³ benchmark to quantify both new-task acquisition and old-task forgetting, with a controlled ablation suite that isolates the marginal contribution of each component. `[TODO: results — headline forgetting/CL-Score numbers vs. CLEAR and pure-RL baselines]`

### Contributions

Framed under the single thesis, our contributions are:

1. **A source-level reframing of continual agentic RL.** We identify and formalize two corruptions of the GRPO learning signal—within-group environment divergence and multi-turn *premise drift*—that arise *before* consolidation and cannot be fixed by replay or regularization. (§3)
2. **A clean-signal data/rollout system.** Winner-synchronized session scheduling restores bit-identical within-group starts across multi-query sessions; a three-agent (observer / questioner / reward) simulated-user pipeline constructs premise-grounded multi-turn data online, doubling as a self-adaptive curriculum that tracks the policy's capability frontier. (§4.4–§4.5)
3. **A consolidation method matched to the cleaned signal.** A four-term CL objective with zero-fork injection, a capability-partitioned anti-forgetting replay buffer, and a U-shaped block-level replay reweighting. (§4.1–§4.3)
4. **A large-scale validation.** A 27B agent over seven domains with controlled per-component ablations on a Pass³ agentic benchmark. `[TODO: results]` (§5–§6)

> **Note on framing discipline.** Contributions 2 and 3 are deliberately *not* presented as a list of independent tricks; each is a means toward the single thesis of producing-then-consolidating an uncorrupted signal. In the ablations (§5) every sub-component must earn its place against this thesis; any that does not improve forgetting under the clean-signal regime is reported as a negative result rather than defended.
