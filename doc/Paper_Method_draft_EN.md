# Paper Draft — Method Section (§4), English

> **Use**: English prose draft of the paper's Method section; drop directly into the paper `.tex`/`.md` and polish.
> **Conventions**: experiment-dependent numbers are marked `[TODO: fill after training]`; equations/hyperparameters follow [`CL_Update_Sunhao.md`](CL_Update_Sunhao.md); citation tags `[A2]`/`[B4]` match its reference list.
> **Chinese version**: [`Paper_Method_draft_CN.md`](Paper_Method_draft_CN.md).
> **Date**: 2026-06-12

---

## 4 Method

We target *continual reinforcement learning of agentic LLMs*: a deployed policy $\pi_\theta$ is periodically updated with freshly collected interaction data, and must acquire new behaviors **without forgetting** previously mastered capabilities. Our method is organized around a single observation that ties together its two halves. Group-relative policy optimization (GRPO) estimates a per-sample advantage by normalizing rewards **within a group of $M$ rollouts of the same query**,
$$A_i = \frac{r_i - \mathrm{mean}(r)}{\mathrm{std}(r) + \epsilon}.$$
This estimator is unbiased *only* when the $M$ trajectories differ purely because of stochastic policy sampling. Any other source of variance—an unequal execution environment across the group, or a multi-turn query whose premise does not actually hold in the realized state—injects environment noise into $A_i$ and corrupts the learning signal at its source.

We therefore decompose the problem into two coupled parts. **First, we produce a clean signal** (§4.4–§4.5): a sandboxed rollout scheduler enforces bit-identical starting states across each group and re-aligns the session to the winning trajectory at every turn boundary, while a simulated-user pipeline constructs multi-turn follow-up queries online so that every follow-up's premise is grounded by construction. **Second, we consolidate that signal without forgetting** (§4.1–§4.3): a four-term continual-learning objective injects experience replay and policy regularization into GRPO, a capability-partitioned replay buffer prioritizes trajectories by their *anti-forgetting value* rather than their reward, and a U-shaped block-level reweighting scheme directs replay credit to the tokens that matter most. Figure 1 gives an overview.

> **Figure 1.** *System overview.* Left: the sandbox rollout scheduler (16 sessions × 8 slots) with per-turn winner synchronization and the three-agent (observe / ask / reward) multi-turn construction loop, producing buffer-ready trajectories with a clean GRPO advantage. Right: the continual-learning trainer consuming online rollouts and replayed trajectories under the $L_{cl}$ objective, with the 7-bucket prioritized buffer in between. A dashed arrow labeled "clean advantage" connects the two halves.

---

### 4.1 Continual-Learning Objective

We optimize a four-term objective that augments the standard RL loss with replay and regularization:
$$L_{cl} = \lambda_1 L_{rl} + \lambda_2 L_{kl} + \lambda_3 L_{replay} + \lambda_4 L_{ent}.$$

$L_{rl}$ is the GRPO loss on freshly rolled-out trajectories and is the primary objective ($\lambda_1=1$ throughout). $L_{kl}$ constrains policy drift via a **reverse** KL divergence to a reference policy, $L_{kl}=D_{KL}(\pi_{new}\,\|\,\pi_{ref})$; whether $\pi_{ref}$ should be the initial policy $\pi_0$ or the previous-stage checkpoint $\pi_{t-1}$ is an empirical question we study in our ablations (§5). $L_{replay}$ is a behavior-cloning term over trajectories sampled from the replay buffer,
$$L_{replay} = \mathbb{E}_{(s,a)\sim \mathcal{B}}\big[-\log \pi_{new}(a\mid s)\cdot w\big],$$
where the per-token weight $w$ is defined in §4.3. $L_{ent}$ is an entropy bonus on online-rollout states that we keep **on by default in every configuration** ($\lambda_4=0.001$).

Two design choices are worth making explicit.

**We drop parameter-space regularization.** A common anti-forgetting term penalizes weight movement, $L_{reg}=\|\theta-\theta_{prev}\|^2$. We set its weight to zero: parameter distance is a poor proxy for functional distance, and $L_{kl}$ constrains the policy directly in output-distribution space, which is both more precise and sufficient. The freed coefficient slot is reallocated to $L_{ent}$.

**Entropy regularization is not optional in this setting.** Multi-turn agentic RL is prone to *Echo Trap* [B4], a collapse of policy diversity in which the $M$ within-group trajectories become near-identical, driving $A_i \to 0$ and stalling learning. $L_{rl}$ and $L_{replay}$ are both mode-seeking and *accelerate* entropy decay, and $L_{kl}$ only keeps $\pi_{new}$ close in shape to $\pi_{ref}$ without preventing entropy collapse. We therefore enable $L_{ent}$ uniformly, including in the no-CL baseline; disabling it there would cause the baseline to collapse and would conflate "forgetting under pure RL" with "degradation after collapse," breaking comparability across configurations.

The objective is injected into the verl/GRPO trainer through its single supported hook (`actor.set_loss_fn`) with **no framework fork**: replayed rows are concatenated into the training batch with a dedicated mask so they contribute only to $L_{replay}$ and never to the PPO denominator.

---

### 4.2 Capability-Partitioned Prioritized Replay Buffer

The replay buffer is the memory that $L_{replay}$ draws from. Two principles distinguish it from a vanilla experience-replay buffer such as CLEAR [A2].

**Partition by capability, not by difficulty.** We organize the buffer into $B=7$ buckets aligned with capability domains (Workflow, SysOps, Dialogue, Finance, Communication, Knowledge/Analysis, OfficeQA). Difficulty is an unstable partition axis—as the policy improves, the difficulty of a task drifts, forcing continual re-classification—whereas capability is stable. Each bucket receives a quota under a square-root, floor-protected allocation,
$$q_i = q_{min} + (C - B\,q_{min})\cdot \frac{n_i^{0.5}}{\sum_j n_j^{0.5}},$$
where $n_i$ is the number of tasks in bucket $i$ and $C$ is the total capacity. The square-root exponent gives larger buckets more room without letting them grow proportionally, and the hard floor $q_{min}$ prevents small but distinct capabilities (e.g., OfficeQA) from being squeezed out. Crucially, **eviction is bucket-local**: a full bucket only evicts its own lowest-priority trajectory, and new tasks can never evict trajectories from another bucket. This is the mechanism that makes the "partition by capability" claim operational—capabilities cannot cannibalize one another.

**Prioritize by anti-forgetting value, not by reward.** Within a bucket, trajectories are retained by a priority that combines four anti-forgetting signals,
$$\text{priority}_i = f\big(\text{forgetting\_risk}_i,\ \text{rarity}_i,\ \text{diversity}_i,\ \text{difficulty}_i\big),$$
with forgetting risk (how much the current policy has regressed on the trajectory) weighted most heavily. We deliberately avoid ranking by absolute reward: as training proceeds and rewards rise globally, a reward-ranked buffer would systematically evict older trajectories and degenerate into a sliding window, defeating the purpose of continual replay. Sampling is two-level—first a bucket (a mix of quota-proportional and uniform selection, with a starvation boost for long-idle buckets), then a trajectory within the bucket by priority-weighted sampling rather than top-$k$ selection, so that replay does not repeatedly reinforce a few "star" trajectories. The buffer module is implemented independently of the RL framework and is therefore unit-testable in isolation.

> **Figure 2.** *Seven-bucket structure with per-bucket quotas.* A tree showing the 7 capability buckets and their task counts, annotated with the square-root quota allocation under a 25k total capacity.

---

### 4.3 U-Shaped Block Reweighting for Replay

The replay weight $w$ in $L_{replay}$ combines a trajectory-level term with a token-level term. Concretely, for token $t$ of trajectory $i$,
$$w_t^{(i)} = \mathrm{normalize}\Big(\mathrm{clip}\big(\text{priority}_i \cdot \tfrac{\gamma^{\,\mathrm{block}(t)} + \delta^{\,K_i - 1 - \mathrm{block}(t)}}{2},\ q_5,\ q_{95}\big)\Big),$$
where $\text{priority}_i$ is the trajectory priority from §4.2 and the second factor is a **U-shaped block weight** applied only to response tokens.

**Action-block segmentation.** Rather than treating all tokens uniformly or decaying per raw token position, we segment each trajectory into $K_i$ *action blocks* delimited by its structural tags (`<think>`, `<toolcall>`, `<observation>`, `<final_answer>`). $K_i$ varies per trajectory; tokens within a block share a weight. When a trajectory cannot be parsed into blocks (or yields a single block), we fall back to equal-length segmentation. This aligns the weight profile with semantic boundaries of the agent loop instead of arbitrary token offsets.

**Why a U shape.** The block weight assigns high weight to both the first block (the early, high-variance branching decision—which tool to call, which path to take) and the last blocks (the conclusion-generation segment), while down-weighting the middle execution detail. An earlier version of this scheme used a monotone decay with a single boost on the final-answer span; we revised it to the symmetric U shape because the importance near the end of a trajectory is not confined to the final-answer tokens—the several blocks approaching the end (pre-conclusion summaries, key judgments) also matter, and a single-point boost cannot capture an entire region. With $\gamma=\delta$, the scheme is continuously tunable: setting $\gamma=\delta=1$ makes every block weight identically $1$, recovering uniform weighting as a built-in baseline, so the only variable separating the uniform and U-shaped configurations is the value of $(\gamma,\delta)$. Finally, we clip the *product* (not the priority alone) to its $[5\%,95\%]$ quantiles before normalization, because extreme weights arise from the interaction of priority and block position and only clipping the final product controls them.

> **Figure 3.** *U-shaped block weight.* Weight vs. normalized block position $\mathrm{block}(t)/K_i$ for $\gamma=\delta\in\{1.0, 0.92, 0.88\}$, showing the flat (uniform) curve at $1.0$ and increasingly pronounced U shapes as $\gamma$ decreases.

---

### 4.4 Sandboxed Rollout with Winner-Synchronized Sessions

We now turn to producing the clean learning signal. Each agentic task is executed in a cloud sandbox: policy inference (the 27B forward pass, which also yields the per-token log-probabilities needed for training) runs **outside** the sandbox on the GPU cluster, while actions (installing packages, writing files, running commands) run **inside** the sandbox, where state is process-snapshottable and hundreds of instances can run concurrently in isolation. Trajectories are collected from the rollout engine's **native** outputs (token ids, response mask, behavior log-probs) rather than reconstructed by an external proxy.

A rollout step runs $16$ sessions in parallel, each with a group of $M=8$ slots, for $128$ concurrent instances. Within a session, every query is handled as follows: the $8$ slots are derived from a single master template so they start **bit-identical**; the policy then rolls out $8$ trajectories that are allowed to diverge during execution (this divergence is the intended source of advantage variance); a winner is selected; and the disk state **and** the conversation history of all $8$ slots are re-aligned to the winner before the next query. The session is discarded at the end without writing back to the master.

This per-turn winner synchronization is what keeps the GRPO advantage clean across a multi-query session. Without it, the $8$ slots of query $k{+}1$ would start from different states, so the within-group comparison would mix policy differences with historical-path differences; a slot could be penalized not for a worse policy but for an unfavorable state left by query $k$, corrupting credit assignment, and the group standard deviation would absorb environment variance that accumulates with query index. We synchronize **only at query boundaries**; mid-query divergence is preserved. One implementation constraint is that the winner instance must stay alive within a session—it is the sole live carrier of accumulated process and in-memory state—so only the seven losing slots are killed and replaced by forks of the live winner.

> **Figure 4.** *Winner-synchronized session.* A timeline for one session: spawn 8 bit-identical slots from the master; for each query, 8 parallel rollouts → select winner → sync all slots (disk + history) to the winner → next query; destroy at session end.

---

### 4.5 Online Multi-Turn Query Construction via Simulated Users

The second source of signal corruption is multi-turn data. A follow-up query typically references the result of the previous turn (e.g., "the figures on page 3 of that deck are wrong"). When such a follow-up is fixed at data-collection time but the realized state $e_k$ is a stochastic function of the policy, the follow-up's premise holds only with probability $\Pr[\phi(q_{k+1})(e_k)] < 1$, and this probability *drifts* as the policy improves. A failed premise produces a corrupted signal: the model is either rewarded for hallucinating a fix to a non-existent problem, or penalized for honestly reporting that no problem exists.

We eliminate premise drift by **constructing follow-ups online, after observing the realized state**. From each real session we retain only the first query as a seed (preserving the true distribution of user intents) and generate subsequent turns at the winner-synchronization boundary with three cooperating agents:

- an **observer** (no persona) that reads the winning actor's output, decides what is relevant to the task—especially intermediate and final results—and actively gathers a neutral, structured report $R_t = \mathrm{Obs}(a_t^w, e_t^w)$ from the live winner sandbox (read-only);
- a **questioner** that carries one of $16$ fixed personas (profession, preference, user profile, and an *observation focus*: whole vs. detail, form vs. content). **Exactly one persona is drawn at the start of each session and held fixed for the entire session—one persona per session, shared by all turns ($p$ is constant within a session)**; the questioner produces the next query $q_{t+1}\sim Q(\cdot\mid p, R_t, H_t)$ where $H_t$ is the winner-derived session history; and
- a **reward model** that scores the turn from the same report, $r_t = \mathrm{Reward}(R_t, a_t^w, \text{rubric})$.

The report $R_t$ serves both downstream consumers, which guarantees that the facts used to score a turn and the facts used to pose the next question are identical, and decouples objective evidence-gathering (observer) from subjective stance (questioner). Because the follow-up is written after the observer has inspected $e_t^w$, its premise is grounded by construction, $\Pr[\phi(q_{t+1})(e_t^w)]\approx 1$. Grounding the reward in the observed *effect* (whether a file was actually produced, whether its contents are correct) rather than the actor's textual claims also guards against reward hacking. **The purpose of the 16 diverse personas is precisely to prevent mode collapse**: a persona-free or single-persona questioner quickly converges to a few templated follow-ups ("please optimize it further"), whose entropy decays with turn count and loses training value; conditioning on personas that differ in profession, preference, and observation focus elicits different facets from the same objective report and keeps the follow-up distribution broad. Complementing this, follow-up length is kept small (sampled in $\{1,2,3\}$), the persona is **resampled only across sessions (fixed within a session)**, and—because each turn's winner state has been altered by the previous follow-up—the report that one fixed persona sees still changes from turn to turn (within-session diversity comes from state evolution, not from switching personas).

**Patience-governed handling of failed turns.** Rather than imposing a fixed retry cap, we make the decision to retry or abandon a failed turn an attribute of the simulated user, governed by two per-persona quantities read in when the persona is loaded at session start: an initial patience $P_0(p)$ and a base decrement $d_0(p)$. Let $k=1,2,\dots$ index consecutive failed turns within the session—a turn fails when its winning response errors, halts early, or leaves the task incomplete. On the $k$-th failure the patience decays by a geometrically growing decrement,
$$P_k = P_{k-1} - d_0(p)\,2^{\,k-1} \;=\; P_0(p) - d_0(p)\,(2^{k}-1).$$
The controller then retries the turn (the questioner issues a "not finished / please redo" follow-up) with probability $\mathrm{clip}(P_k, 0, 1)$ and otherwise ends the session with `<end_session>`; in particular $P_k<0$ deterministically terminates. Making *both* $P_0$ and $d_0$ persona-specific decouples two distinct traits—*initial tolerance* (whether the user grants a retry at all) and *escalation rate* (how quickly frustration mounts)—so that, e.g., a "polite-but-short-tempered" persona (high $P_0$, high $d_0$) is distinguishable from a "steadily patient" one (high $P_0$, low $d_0$); a global default is used when a persona omits $d_0$, and concrete values are left as `[TODO: fix per persona]`. Because the decrement doubles each time, patience crosses zero after $O(\log_2(P_0/d_0))$ failures, so retries are intrinsically bounded without a separate cap (with $d_0{=}0.1, P_0{\approx}1$ this reproduces the familiar "$\le 3$ retries"). The geometric decay models escalating user frustration; the patience-proportional retry probability lets impatient personas abandon sooner and patient ones persist; and the stochasticity prevents the policy from learning a fixed retry count. Patience is consumed only on the failure path—successful turns proceed under the normal follow-up budget $K$, and redo turns do not count against $K$.

A useful side effect is an **adaptive curriculum**: the questioner always critiques the *current* policy's actual output, so as the policy strengthens and residual flaws become subtler, the follow-ups it raises become correspondingly finer, tracking the capability frontier without a hand-designed difficulty schedule.

> **Figure 5.** *Three-agent multi-turn construction.* At a winner-sync boundary: observer reads the winner trajectory + sandbox → produces report $R_t$; $R_t$ fans out to the reward model (score) and to the persona-conditioned questioner (next query or end-of-session).

---

> *Transition to §5.* Having defined how a clean signal is produced (§4.4–§4.5) and consolidated without forgetting (§4.1–§4.3), we now design controlled ablations that isolate the marginal contribution of each component.

---

## Appendix A: Prompts

> **Note**: This appendix holds the complete prompts for the three agents and the reward model. The paper requires the full prompts to appear at the end; they are **left blank pending implementation** (design items O3/O4/O6 in [`UserSim_多轮Query在线生成.md`](UserSim_多轮Query在线生成.md)). Once written, replace each code block in place; the main text does not change.

### A.1 Observer prompt (O6)

> Role (see §4.5): no persona; driven by the actor's output, collects intermediate and final results, and produces the objective report $R_t$.

```text
[TODO: fill in the Observer system prompt before training]
(to be decided: read-only command set, how to locate artifacts from the actor's
 claims, intermediate-result detection and capture, file-tree depth,
 content-truncation / token budget, structured ObservationReport output format)
```

### A.2 Questioner prompt (O3)

> Role (see §4.5): carries the session-fixed persona $p$, reads report $R_t$ and history $H_t$, emits the next query or `<end_session>`.

```text
[TODO: fill in the Questioner system prompt before training]
(to be decided: how persona fields are injected, how the observation focus shapes
 which facet to ask about, how the patience mechanism on failure (§4.5: geometric
 decay of $P_k$ + clip-probability redo) is surfaced in the prompt,
 `<end_session>` trigger conditions, anti-"AI-ese" instructions, output format)
```

### A.3 Reward prompt / rubric (O4)

> Role (see §4.5): scores the turn from the observation report $R_t$ against the rubric.

```text
[TODO: fill in the reward judging prompt / rubric before training]
(to be decided: input construction and truncation budget for report + trajectory,
 whether to reuse ClawEval's safety×(0.8·completion+0.2·robustness), the criterion
 for "was the user's request fulfilled", output score format)
```
