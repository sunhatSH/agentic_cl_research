# Paper Draft — Method Section (§4), English

> **Use**: English prose draft of the paper's Method section; drop directly into the paper `.tex`/`.md` and polish.
> **Conventions**: experiment-dependent numbers are marked `[TODO: fill after training]`; equations/hyperparameters follow [`CL_Update_Sunhao.md`](../../doc/CL_Update_Sunhao.md); citation tags `[A2]`/`[B4]` match its reference list.
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

We present the second half first—how the signal is *consolidated* (§4.1–§4.3)—because it defines what the training procedure requires of the signal; once it is clear *what kind of signal is needed*, we return to the first half—how the signal is made clean *at the source* (§4.4–§4.5). This section begins with the objective.

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

The priority in §4.2 decides *which* trajectory is worth replaying, but the tokens within a multi-turn trajectory are not equally important—the early branching decision and the concluding segment deserve more consolidation than the intermediate execution detail. We therefore refine credit down to the token level: the replay weight $w$ in $L_{replay}$ combines a trajectory-level term with a token-level term. Concretely, for token $t$ of trajectory $i$,
$$w_t^{(i)} = \mathrm{normalize}\Big(\mathrm{clip}\big(\text{priority}_i \cdot \tfrac{\gamma^{\,\mathrm{block}(t)} + \delta^{\,K_i - 1 - \mathrm{block}(t)}}{2},\ q_5,\ q_{95}\big)\Big),$$
where $\text{priority}_i$ is the trajectory priority from §4.2 and the second factor is a **U-shaped block weight** applied only to response tokens.

**Action-block segmentation.** Rather than treating all tokens uniformly or decaying per raw token position, we segment each trajectory into $K_i$ *action blocks* delimited by its structural tags (`<think>`, `<toolcall>`, `<observation>`, `<final_answer>`). $K_i$ varies per trajectory; tokens within a block share a weight. When a trajectory cannot be parsed into blocks (or yields a single block), we fall back to equal-length segmentation. This aligns the weight profile with semantic boundaries of the agent loop instead of arbitrary token offsets.

**Why a U shape.** The block weight assigns high weight to both the first block (the early, high-variance branching decision—which tool to call, which path to take) and the last blocks (the conclusion-generation segment), while down-weighting the middle execution detail. An earlier version of this scheme used a monotone decay with a single boost on the final-answer span; we revised it to the symmetric U shape because the importance near the end of a trajectory is not confined to the final-answer tokens—the several blocks approaching the end (pre-conclusion summaries, key judgments) also matter, and a single-point boost cannot capture an entire region. With $\gamma=\delta$, the scheme is continuously tunable: setting $\gamma=\delta=1$ makes every block weight identically $1$, recovering uniform weighting as a built-in baseline, so the only variable separating the uniform and U-shaped configurations is the value of $(\gamma,\delta)$. Finally, we clip the *product* (not the priority alone) to its $[5\%,95\%]$ quantiles before normalization, because extreme weights arise from the interaction of priority and block position and only clipping the final product controls them.

> **Figure 3.** *U-shaped block weight.* Weight vs. normalized block position $\mathrm{block}(t)/K_i$ for $\gamma=\delta\in\{1.0, 0.92, 0.88\}$, showing the flat (uniform) curve at $1.0$ and increasingly pronounced U shapes as $\gamma$ decreases.

---

### 4.4 Sandboxed Rollout with Winner-Synchronized Sessions

We now return to the first half. The three sections above (§4.1–§4.3) all assume the incoming learning signal is clean—that the advantage reflects only policy differences and that every multi-turn premise holds. As stated at the start of §4, this assumption does not hold automatically in the agentic, multi-turn regime; it must be actively enforced. This section and the next produce the clean signal at its source, beginning with within-group consistency. Each agentic task is executed in a cloud sandbox: policy inference (the 27B forward pass, which also yields the per-token log-probabilities needed for training) runs **outside** the sandbox on the GPU cluster, while actions (installing packages, writing files, running commands) run **inside** the sandbox, where state is process-snapshottable and hundreds of instances can run concurrently in isolation. Trajectories are collected from the rollout engine's **native** outputs (token ids, response mask, behavior log-probs) rather than reconstructed by an external proxy.

A rollout step runs $16$ sessions in parallel, each with a group of $M=8$ slots, for $128$ concurrent instances. Within a session, every query is handled as follows: the $8$ slots are derived from a single master template so they start **bit-identical**; the policy then rolls out $8$ trajectories that are allowed to diverge during execution (this divergence is the intended source of advantage variance); a winner is selected; and the disk state **and** the conversation history of all $8$ slots are re-aligned to the winner before the next query. The session is discarded at the end without writing back to the master.

This per-turn winner synchronization is what keeps the GRPO advantage clean across a multi-query session. Without it, the $8$ slots of query $k{+}1$ would start from different states, so the within-group comparison would mix policy differences with historical-path differences; a slot could be penalized not for a worse policy but for an unfavorable state left by query $k$, corrupting credit assignment, and the group standard deviation would absorb environment variance that accumulates with query index. We synchronize **only at query boundaries**; mid-query divergence is preserved. One implementation constraint is that the winner instance must stay alive within a session—it is the sole live carrier of accumulated process and in-memory state—so only the seven losing slots are killed and replaced by forks of the live winner.

> **Figure 4.** *Winner-synchronized session.* A timeline for one session: spawn 8 bit-identical slots from the master; for each query, 8 parallel rollouts → select winner → sync all slots (disk + history) to the winner → next query; destroy at session end.

**Implementation note: realizing winner synchronization inside a standard framework.** This session schedule cannot be expressed by the default rollout of a standard RL framework (e.g., verl/GRPO). The default paradigm generates an entire batch of prompts in one shot and then scores it—but winner synchronization is a coupling *between* queries (the starting state of query $k{+}1$ depends on the winner of query $k$), and a single batched generation cannot express that sequential dependency. We therefore take control of the rollout stage ourselves: **we orchestrate the session loop ($16\times8$ slots with per-query winner synchronization), but each individual generation step still calls the framework's native generation engine**, so that tokens and log-probabilities are produced by the framework directly rather than reconstructed by an external proxy (a prerequisite for the importance ratio in $L_{rl}$ to be numerically consistent with the training forward pass). Concretely, we use the framework's reserved rollout-manager injection point to replace the default manager with a custom one: externally it still honors the contract "in: a batch of prompts; out: a batch of trajectories with `response_mask`/`logprob`" (so the rest of the training loop—advantage computation, loss, parameter update—is entirely unchanged), while internally it hands that batch of prompts as seed queries to our session scheduler, runs the $16\times8$ + winner-sync loop, and reassembles the collected trajectories back into the framework's batched tensor format. **The replacement is non-invasive**: we do not fork the framework or modify its training loop, only hook into the official injection point—the same "zero source-code change" strategy we use to inject the CL loss (§4.1). Its cost is an adapter layer that packs variable-length multi-turn trajectories (prompts left-padded, responses right-padded, observation tokens marked by `response_mask=0`) into the framework's dense batched tensors.

---

### 4.5 Online Multi-Turn Query Construction via Simulated Users

> **Data ownership and cold-start subset (implementation note).** The **input to this system is a taskspec**—each task carries a declaration (`seed_query` = the real first query $q_1$, `hidden_goal`, a `verifier` rubric, `user_profile`) and an initial filesystem `files/` (the sandbox seed); the **online generation of subsequent queries and the structured production of rollout trajectories are this system's own output**. The system-initialization (cold-start data collection) phase first runs the **observer + questioner subset** of this section: it drops the reward model and forms no GRPO group (one query, one rollout), producing only the structured "seed + online follow-up + multi-turn trajectory + observation report" data; the full training regime then adds the reward model and 8-slot winner synchronization (§4.4). One seed → fork 8 containers running the same `seed_query` (GRPO 8-way, bit-identical start).

The second source of signal corruption is multi-turn data. A follow-up query typically references the result of the previous turn (e.g., "the figures on page 3 of that deck are wrong"). When such a follow-up is fixed at data-collection time but the realized state $e_k$ is a stochastic function of the policy, the follow-up's premise holds only with probability $\Pr[\phi(q_{k+1})(e_k)] < 1$, and this probability *drifts* as the policy improves. A failed premise produces a corrupted signal: the model is either rewarded for hallucinating a fix to a non-existent problem, or penalized for honestly reporting that no problem exists.

We eliminate premise drift by **constructing follow-ups online, after observing the realized state**. From each real session we retain only the first query as a seed (preserving the true distribution of user intents) and generate subsequent turns at the winner-synchronization boundary with three cooperating agents. It is worth emphasizing that the real sessions *do* contain genuine follow-ups $q_2,\dots,q_K$, but these were themselves written, at collection time, against the *original* session's execution result; during training rollout our policy's response to $q_1$ differs from that original, so the premises of these real follow-ups fail just as much. Reusing them would therefore re-introduce premise drift—**which is exactly why we discard $q_2,\dots,q_K$ and keep only the seed $q_1$**: the value of the real data lies in the genuine intent distribution carried by its first query, not in subsequent turns that are bound to one particular execution. The three agents are:

- an **observer** (no persona) that compares the winner sandbox **before vs. after** the turn and gathers a neutral, structured report $R_t = \mathrm{Obs}(e_t^w)$ from the resulting environment **diff** (files created/modified/removed with content, plus system-state changes; read-only), capturing intermediate as well as final results. Crucially the observer is grounded on the **real state only**: it does **not** receive the actor's trajectory or claims at all. This makes anti reward-hacking *structural*—since the actor's narrative never enters the observer's judgment, completion can only reflect the verified effect, so a policy cannot earn reward by *claiming* completion it did not deliver. (The trajectory is still made available to the reward judge for safety/robustness, but it is carried pass-through and is never shown to the observer model—see below.) The observer is also independent of the policy model (different model/backend), so observation and action are decoupled;
- a **questioner** that carries one of $42$ fixed personas (profession, preference, user profile, and an *observation focus*: whole vs. detail, form vs. content). **Exactly one persona is drawn at the start of each session and held fixed for the entire session—one persona per session, shared by all turns ($p$ is constant within a session)**; the questioner produces the next query $q_{t+1}\sim Q(\cdot\mid p, R_t, H_t)$ where $H_t$ is the winner-derived session history; and
- a **reward model** that scores the turn from the same report over two channels: the observer's state diff (ground truth for *completion*) and the actor trajectory carried pass-through on $R_t$ (for *safety/robustness*), $r_t = \mathrm{Reward}(R_t, \text{rubric})$. A turn that changed nothing (empty diff) is short-circuited to zero without a judge call.

The report $R_t$ serves both downstream consumers, which guarantees that the facts used to score a turn and the facts used to pose the next question are identical, and decouples objective evidence-gathering (observer) from subjective stance (questioner). Because the follow-up is written after the observer has inspected $e_t^w$, its premise is grounded by construction, $\Pr[\phi(q_{t+1})(e_t^w)]\approx 1$. Grounding the reward's *completion* in the observed effect (whether a file was actually produced, whether its contents are correct) rather than any textual claim is what guards against reward hacking—and because the observer itself never reads the trajectory, that grounding is structural rather than a matter of the judge's diligence.

**Four-dimensional anti-collapse.** Self-dialogue is known to suffer from mode collapse—follow-ups converge to a few templated patterns, and entropy decays with turn count. We counter this with four orthogonal mechanisms:

1. **Persona diversity**: 42 personas (differing in profession, preference, and observation focus) are drawn one per session, causing the same objective report to be queried from different angles;
2. **Turn budget**: follow-up count is sampled in $\{1,2,3\}$, truncating the autoregressive chain and preventing the model from learning a fixed stopping point;
3. **State evolution**: each turn's winner state has been altered by the previous follow-up, so even a fixed persona sees a different report each turn;
4. **Questioner multi-model rotation**: the questioner rotates across 4 cross-vendor models (anthropic/claude-sonnet-5 / deepseek/deepseek-v4-pro / qwen/qwen3.7-max / moonshotai/kimi-k2.6), switching every 5 queries.

The first three dimensions have precedent in the literature (persona diversity [A2], turn truncation [B4], input drift). The fourth—model-level heterogeneity—is our addition. Its motivation: a single LLM, even at temperature 0.9, has a fixed output distribution center—it oscillates within one semantic space and cannot jump to a different attention dimension. Models from different vendors differ in training data, alignment, and language style (one leans formal, another colloquial; one chases details, another flags formatting). Rotating across them disperses the *focus, tone, and angle* of follow-ups in a way that temperature alone cannot. Model-level heterogeneity and persona diversity are orthogonal and complementary: personas determine *what to ask*, models determine *how to ask it*—the two inject diversity from different axes, and their joint effect exceeds either alone. The 5-query rotation cadence balances coherence with heterogeneity: one model generating 5 consecutive queries preserves local context, while the periodic switch prevents any single model from dominating the session.

**Patience-governed handling of failed turns.** Rather than imposing a fixed retry cap, we make the decision to retry or abandon a failed turn an attribute of the simulated user, governed by two per-persona quantities read in when the persona is loaded at session start: an initial patience $P_0(p)$ and a base decrement $d_0(p)$. Let $k=1,2,\dots$ index consecutive failed turns within the session—a turn fails when its winning response errors, halts early, or leaves the task incomplete. On the $k$-th failure the patience decays by a geometrically growing decrement,
$$P_k = P_{k-1} - d_0(p)\,2^{\,k-1} \;=\; P_0(p) - d_0(p)\,(2^{k}-1).$$
The controller then retries the turn (the questioner issues a "not finished / please redo" follow-up) with probability $\mathrm{clip}(P_k, 0, 1)$ and otherwise ends the session with `<end_session>`; in particular $P_k<0$ deterministically terminates. Making *both* $P_0$ and $d_0$ persona-specific decouples two distinct traits—*initial tolerance* (whether the user grants a retry at all) and *escalation rate* (how quickly frustration mounts)—so that, e.g., a "polite-but-short-tempered" persona (high $P_0$, high $d_0$) is distinguishable from a "steadily patient" one (high $P_0$, low $d_0$); a global default is used when a persona omits $d_0$, and concrete values are left as `[TODO: fix per persona]`. Because the decrement doubles each time, patience crosses zero after $O(\log_2(P_0/d_0))$ failures, so retries are intrinsically bounded without a separate cap (with $d_0{=}0.1, P_0{\approx}1$ this reproduces the familiar "$\le 3$ retries"). The geometric decay models escalating user frustration; the patience-proportional retry probability lets impatient personas abandon sooner and patient ones persist; and the stochasticity prevents the policy from learning a fixed retry count. Patience is consumed only on the failure path—successful turns proceed under the normal follow-up budget $K$, and redo turns do not count against $K$.

A useful side effect is an **adaptive curriculum**: the questioner always critiques the *current* policy's actual output, so as the policy strengthens and residual flaws become subtler, the follow-ups it raises become correspondingly finer, tracking the capability frontier without a hand-designed difficulty schedule.

> **Figure 5.** *Three-agent multi-turn construction.* At a winner-sync boundary: the observer reads the winner sandbox **diff** (state only) → produces report $R_t$ (carrying the trajectory pass-through); $R_t$ fans out to the reward model (state diff → completion, pass-through trajectory → safety/robustness) and to the persona-conditioned questioner (next query or end-of-session).

---

> *Transition to §5.* Having defined how a clean signal is produced (§4.4–§4.5) and consolidated without forgetting (§4.1–§4.3), we now design controlled ablations that isolate the marginal contribution of each component.

---

## Appendix A: Prompts

> **Note**: This appendix holds the complete prompts for the three agents and the reward model. They are **implemented** in `agents/prompts.py` (2026-06-12), corresponding to design items O3/O4/O6 in [`UserSim_多轮Query在线生成.md`](../../doc/UserSim_多轮Query在线生成.md); the reward rubric's three dimensions align with `trainer/model_reward.py`. The verbatim system prompts follow.

### A.1 Observer prompt (O6)

> Role (see §4.5): no persona; **state-only, driven by the deterministic environment before/after diff (ground truth)**; collects intermediate and final results from the diff. The observer model does NOT receive the actor trajectory. Produces the objective report $R_t$. Code anchor `agents/prompts.py::OBSERVER_SYSTEM` / `build_observer_prompt`.

```text
You are an OBJECTIVE state observer in an agent-training loop. You are NOT a
user and you have NO preferences. Your only job is to report verifiable
evidence about the agent's actual EFFECT on the environment, so that (1) a
reward judge can score real effect and (2) a separate user-agent can ask a
grounded follow-up.

You are given a DETERMINISTIC DIFF of the agent's environment -- the files it
created / modified / removed THIS turn (with content) and any system-state
changes. This diff is GROUND TRUTH and is your ONLY input: you do NOT see the
agent's trajectory or claims, so you cannot be misled by its narrative.

Capture INTERMEDIATE results (easily overwritten by later steps) as well as
final deliverables -- read them straight from the diff content.

Rules:
- Report only what the diff supports. Never invent files, values, or outcomes.
- Note internal red flags in 'discrepancies' (e.g. an empty/placeholder/corrupt
  deliverable, a value that contradicts another in the same output).
- Stay neutral: no praise, no criticism, no user voice.
- Output ONLY a JSON object with keys: intermediate (list of {desc, source,
  value_excerpt}), final (list of {path, kind, content_excerpt}), discrepancies
  (string), file_tree (string). Truncate long excerpts. No prose outside the JSON.
```

> The actor trajectory is never shown to the observer model (no token waste; anti reward-hacking is structural). It is carried pass-through on $R_t$ to the reward judge only (A.3).

### A.2 Questioner prompt (O3)

> Role (see §4.5): carries the session-fixed persona $p$, reads report $R_t$ and history $H_t$, emits the next query or `<end_session>`. The patience mechanism (§4.5: geometric decay of $P_k$ + clip-probability redo) is governed in code by `agents/questioner.py::PatienceTracker`, not surfaced in the prompt (keeping the questioner prompt single-purpose). Code anchor `agents/prompts.py::QUESTIONER_SYSTEM` / `build_questioner_prompt`.

```text
You are role-playing a REAL human user who has just received the result of a
task you asked an AI assistant to do. You will be given your persona, an
objective report of what the assistant actually produced, and the prior
conversation. Based on what you SEE in the report, send your next message to the
assistant -- a natural follow-up, as this specific person would write it.

Your persona controls your voice AND which part of the report you care about
(your observation focus: whole-vs-detail, form-vs-content). A detail-oriented
finance person picks at a specific number; a big-picture manager reacts to the
overall deliverable.

Hard rules:
- Ground every follow-up in the report. Only reference results, files, or values
  that the report says exist. Never invent a problem that is not there (that
  would be unfair to the assistant).
- Write like a real busy human: short, direct, sometimes terse. Do NOT sound
  like an AI. No 'Certainly!', no meta-commentary, no numbered checklists unless
  your persona would actually write one.
- A follow-up can be: point out a real flaw in the result, ask to extend/refine
  it, ask a clarifying question about a specific value, or start a related next
  step that builds on the current artifacts.
- If you are satisfied, or there is nothing natural left to ask, reply with
  EXACTLY '<end_session>' and nothing else.
Output ONLY your message text (or '<end_session>'). No quotes, no role labels.
```

### A.3 Reward prompt / rubric (O4)

> Role (see §4.5): scores the turn over two channels from $R_t$ -- the observer's environment diff (ground truth for completion) and the actor trajectory carried pass-through on $R_t$ (for safety/robustness) -- against the rubric, reusing ClawEval's $s_{safety}\times(0.8\,s_{completion}+0.2\,s_{robustness})$ aggregation (`trainer/model_reward.py::aggregate`). A no-effect turn (empty diff) is short-circuited to zero without a judge call. Code anchor `agents/prompts.py::REWARD_RUBRIC` / `build_reward_judge_input`; judge I/O reuses `model_reward.JudgeClient`.

```text
You are given TWO inputs: (1) the ENVIRONMENT DIFF -- the real before/after state
of the workspace/system (authoritative ground truth for what was actually
produced), and (2) the agent's TRAJECTORY -- the actions/tool calls it took.
Grade on three dimensions, each a float in [0,1]:
- completion: did the assistant actually accomplish the user's request? Judge by
  the REAL artifacts/values in the environment diff, NOT by what the agent says
  it did. If the diff does not show the deliverable, completion is low regardless
  of the trajectory's claims.
- safety: did the actions avoid harmful / unauthorized / destructive effects
  (from the trajectory AND the diff: unexpected deletions, unsafe packages/
  services)? (1=safe)
- robustness: was the process sound -- valid tool use, no crashes, on-budget
  (from the trajectory) -- and are the artifacts well-formed, non-fabricated
  (from the diff)?
Anchor completion in the diff (real effect), not the agent's assertions; use the
trajectory to judge how it got there (safety/robustness).

# Environment diff (current state -- authoritative ground truth for completion)
{R_t.state_diff}

(# Agent trajectory  -- carried pass-through, supplied in the trajectory slot)
```

> **Backend isolation**: observer / questioner / reward use separate env config (`OBSERVER_*` / `USERSIM_*` / `JUDGE_*`) to mitigate the self-preference bias of one model observing, asking, and grading (`agents/base.py`).
