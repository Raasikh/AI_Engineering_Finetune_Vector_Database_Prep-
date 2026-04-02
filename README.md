# Fine-Tuning, Model Adaptation, Vector Databases & Embeddings — Interview Prep

---

## Part 1: Fine-Tuning and Model Adaptation

---

### 1. What is fine-tuning, and when should you fine-tune an LLM?

**Fine-tuning** is the process of taking a pre-trained LLM and continuing its training on a smaller, task-specific or domain-specific dataset so that the model adapts its behavior, style, or knowledge to better serve a particular use case.

**When to fine-tune:**

- **Style/format control**: You need outputs in a very specific format, tone, or structure that prompt engineering cannot reliably achieve (e.g., always outputting valid JSON, always responding in a clinical tone).
- **Domain specialization**: You have a large corpus of domain-specific knowledge (legal contracts, medical records, financial filings) and need the model to internalize domain jargon, reasoning patterns, and conventions.
- **Latency/cost optimization**: Prompt engineering requires very long system prompts or many few-shot examples that inflate token usage. Fine-tuning "bakes in" those patterns, reducing inference cost.
- **Behavioral alignment**: You need the model to consistently follow safety guidelines, refuse certain queries, or adhere to company policies.
- **Task-specific performance**: Classification, extraction, summarization in a narrow domain where a general model underperforms.

**When NOT to fine-tune:**

- If prompt engineering or RAG solves the problem (cheaper, faster iteration).
- If you lack sufficient high-quality training data (hundreds to thousands of examples minimum).
- If the knowledge changes frequently (RAG is better for dynamic knowledge).

---

### 2. Explain the difference between full fine-tuning and parameter-efficient fine-tuning (PEFT).

| Aspect | Full Fine-Tuning | PEFT |
|---|---|---|
| **Parameters updated** | All model parameters (billions) | A small subset or added parameters (0.01–1% of total) |
| **GPU memory** | Very high — must store all parameters, gradients, and optimizer states | Significantly lower — frozen base model + small trainable component |
| **Risk of catastrophic forgetting** | Higher — all weights shift | Lower — base weights are frozen |
| **Training data needed** | Larger datasets preferred | Can work with smaller datasets |
| **Storage** | Full copy of the model per task | Only the small adapter/delta per task (MBs vs GBs) |
| **Techniques** | Standard backprop through all layers | LoRA, QLoRA, Prefix Tuning, Prompt Tuning, Adapters |
| **Multi-task serving** | Separate model per task | Swap adapters on the same base model |

**Full fine-tuning** updates every weight in the model. It can achieve the highest performance ceiling but requires enormous compute, risks destroying pre-trained knowledge, and produces a full-sized model copy for each task.

**PEFT** freezes the vast majority of parameters and only trains a small number of new or modified parameters. This is more practical, cheaper, and allows serving multiple tasks by hot-swapping lightweight adapters on a shared base model.

---

### 3. What is LoRA (Low-Rank Adaptation), and how does it work?

**LoRA** freezes the pre-trained weight matrices and injects trainable low-rank decomposition matrices into each transformer layer.

**How it works:**

For a pre-trained weight matrix `W ∈ R^(d×k)`, instead of learning a full update `ΔW`, LoRA decomposes it:

```
ΔW = B × A
```

Where:
- `A ∈ R^(r×k)` — a down-projection matrix (initialized randomly, often Gaussian)
- `B ∈ R^(d×r)` — an up-projection matrix (initialized to zeros so ΔW = 0 at start)
- `r` is the **rank** (typically 4–64), much smaller than `d` or `k`

During forward pass: `h = Wx + BAx`

**Why it works:**

- Research shows that weight updates during fine-tuning have low intrinsic rank — you don't need to update all dimensions.
- Only `A` and `B` are trained → drastically fewer parameters (e.g., r=16 on a 4096×4096 matrix: 2 × 4096 × 16 = 131K params vs. 16.7M).
- At inference, `BA` can be **merged** into `W` (i.e., `W' = W + BA`), adding zero latency.

**Typical application:** Applied to the query (`Wq`) and value (`Wv`) projection matrices in self-attention, though it can also be applied to `Wk`, `Wo`, and FFN layers.

---

### 4. What is QLoRA, and how does it enable fine-tuning on consumer hardware?

**QLoRA** (Quantized LoRA) combines three innovations to enable fine-tuning of very large models on a single GPU:

1. **4-bit NormalFloat (NF4) Quantization**: The base model weights are quantized to 4-bit precision using a data type optimized for normally distributed weights. This reduces model memory by ~4x (e.g., a 65B model from ~130GB FP16 to ~33GB in 4-bit).

2. **Double Quantization**: The quantization constants themselves are quantized (quantizing the quantization scales), saving an additional ~0.37 bits per parameter.

3. **Paged Optimizers**: Uses NVIDIA unified memory to page optimizer states between CPU and GPU, preventing out-of-memory errors during gradient spikes.

**How it works together:**
- The base model stays frozen in 4-bit.
- LoRA adapters are added in higher precision (BF16/FP16).
- During the forward pass, 4-bit weights are dequantized on-the-fly to BF16 for computation.
- Gradients flow only through the small LoRA parameters.
- The optimizer states are only for the LoRA parameters (tiny).

**Impact**: Fine-tune a 65B parameter model on a single 48GB GPU (A6000) with no significant quality loss compared to full 16-bit fine-tuning. A 7B model can be fine-tuned on a consumer GPU with 24GB VRAM.

---

### 5. Explain Prefix Tuning and Prompt Tuning. How are they different from LoRA?

**Prompt Tuning:**
- Prepends a set of **learnable soft token embeddings** (virtual tokens) to the input at the embedding layer only.
- These soft tokens are continuous vectors (not from the vocabulary) and are optimized via backpropagation.
- The rest of the model is completely frozen.
- Only the soft prompt embeddings are trained — extremely parameter-efficient.
- Limitation: influence is limited because the soft tokens only exist at the input level.

**Prefix Tuning:**
- Prepends learnable continuous vectors (prefixes) to the **key and value matrices at every transformer layer**, not just the input.
- This gives the prefix deeper influence over the model's internal representations.
- A small MLP reparameterization network generates the prefix vectors during training for stability, then discarded at inference.
- More expressive than Prompt Tuning because it affects every layer's attention computation.

**Differences from LoRA:**

| Aspect | LoRA | Prefix/Prompt Tuning |
|---|---|---|
| **Where it acts** | Modifies weight matrices (Wq, Wv, etc.) | Adds virtual tokens to input/KV cache |
| **Mechanism** | Low-rank matrix decomposition | Learnable continuous prefix vectors |
| **Inference overhead** | Zero (adapters can be merged into weights) | Non-zero (prefix tokens consume context window / KV cache space) |
| **Context window** | No impact | Reduces effective context by prefix length |
| **Performance** | Generally higher, especially at smaller ranks | Competitive on some tasks, weaker on others |
| **Composability** | Can merge/stack adapters | Prefixes concatenated but can interfere |

---

### 6. What is adapter-based fine-tuning?

**Adapter-based fine-tuning** inserts small trainable neural network modules (adapters) between the existing frozen layers of a pre-trained transformer.

**Architecture:**
- An adapter is typically a **bottleneck MLP**: a down-projection (d → r), a nonlinearity (ReLU/GELU), and an up-projection (r → d), plus a residual/skip connection.
- Adapters are inserted after the self-attention and/or feed-forward sub-layers in each transformer block.
- Only the adapter parameters are trained; all original model weights stay frozen.

**Key properties:**
- Parameter count is controlled by the bottleneck dimension `r`.
- The skip connection ensures that when adapter weights are near zero, the module is close to an identity function (safe initialization).
- Multiple adapters can be trained for different tasks and swapped at inference.

**Comparison to LoRA:** Adapters add sequential computation (extra layers in the forward pass), which introduces some inference latency. LoRA modifies existing matrices and can be merged, adding zero inference cost. LoRA has largely superseded adapters in popularity for this reason.

---

### 7. What is RLHF (Reinforcement Learning from Human Feedback), and how is it used to align LLMs?

**RLHF** is a three-stage process for aligning an LLM's outputs with human preferences:

**Stage 1 — Supervised Fine-Tuning (SFT):**
- Fine-tune the base model on high-quality demonstrations of desired behavior (instruction-response pairs written by humans).
- Produces a model that can follow instructions but may still produce undesirable outputs.

**Stage 2 — Reward Model Training:**
- Collect **comparison data**: for a given prompt, the SFT model generates multiple responses, and human annotators rank them by preference (e.g., A > B > C).
- Train a **reward model** (often initialized from the SFT model with a scalar output head) using a pairwise ranking loss (Bradley-Terry model):
  ```
  Loss = -log(σ(r(preferred) - r(rejected)))
  ```
- The reward model learns to assign higher scores to responses humans prefer.

**Stage 3 — RL Optimization (PPO):**
- Use Proximal Policy Optimization to fine-tune the SFT model to maximize the reward model's scores.
- A KL divergence penalty against the SFT model prevents the policy from drifting too far (reward hacking / mode collapse):
  ```
  Objective = E[R(y|x)] - β · KL(π_RL || π_SFT)
  ```
- The β coefficient controls the tradeoff between reward maximization and staying close to the SFT policy.

**Why it matters:** RLHF produces models that are more helpful, harmless, and honest by optimizing directly for human preference signals rather than just next-token prediction.

---

### 8. What is instruction tuning, and why is it important for chat models?

**Instruction tuning** is supervised fine-tuning on datasets of (instruction, response) pairs that teach the model to follow natural language instructions.

**Process:**
- Curate a diverse dataset of instructions spanning many tasks: summarization, QA, translation, code generation, reasoning, creative writing, etc.
- Each example has an instruction (and optional input context) paired with a desired response.
- Fine-tune the base model on this dataset.

**Why it's important for chat models:**
- **Base models are completion engines**: They predict the next token and may not understand that a question expects an answer. They might continue the question, generate more questions, or produce unstructured text.
- **Instruction tuning bridges the gap**: It teaches the model to interpret user intent and produce structured, relevant responses.
- **Generalization**: Models trained on diverse instructions can generalize to unseen instructions (zero-shot instruction following).
- **Safety and format**: Instruction tuning is the foundation for teaching models to refuse harmful requests, follow system prompts, and output in expected formats (JSON, markdown, etc.).

**Examples of instruction tuning datasets:** FLAN, Alpaca, Dolly, OpenAssistant, ShareGPT.

---

### 9. How do you prepare a dataset for fine-tuning an LLM?

**Step 1 — Define the task and format:**
- Decide on the input-output format (e.g., `{"instruction": "...", "input": "...", "output": "..."}` or chat format with system/user/assistant roles).
- Match the format expected by your training framework (e.g., ChatML, Alpaca format).

**Step 2 — Data collection:**
- **Manual curation**: Domain experts write high-quality examples.
- **Existing data**: Convert logs, documentation, support tickets, etc.
- **Synthetic generation**: Use a strong LLM (GPT-4, Claude) to generate examples, then filter for quality.
- **Public datasets**: Leverage existing instruction datasets as a foundation.

**Step 3 — Data quality:**
- Remove duplicates and near-duplicates (deduplication with MinHash or embedding similarity).
- Filter for correctness (factual accuracy, code that compiles, valid JSON).
- Ensure diversity across task types, difficulty levels, and edge cases.
- Remove PII, toxic content, and copyrighted material.
- Validate formatting consistency.

**Step 4 — Data quantity guidelines:**
- Minimum ~100–500 high-quality examples for LoRA on a narrow task.
- 1,000–10,000+ examples for broader behavioral changes.
- Quality matters more than quantity — 500 excellent examples often beat 50,000 noisy ones.

**Step 5 — Train/validation split:**
- Hold out 5–10% as a validation set (stratified by task type if possible).
- Ensure no data leakage between splits.

**Step 6 — Tokenization and preprocessing:**
- Apply the model's tokenizer and verify sequence lengths.
- Truncate or filter examples exceeding max context length.
- Apply appropriate chat templates and special tokens.

---

### 10. What is catastrophic forgetting, and how do you prevent it during fine-tuning?

**Catastrophic forgetting** is the phenomenon where a neural network, when trained on new data, overwrites the knowledge learned from previous training. In the context of LLMs, a model fine-tuned on domain-specific data may lose its general language understanding, reasoning ability, or knowledge of other domains.

**Prevention strategies:**

1. **PEFT methods (LoRA, adapters):** Freeze the base model weights. Since the pre-trained parameters don't change, general knowledge is preserved by design.

2. **Low learning rate:** Use a small learning rate (1e-5 to 5e-5 for full fine-tuning) to make gradual updates rather than large weight shifts.

3. **Data mixing / replay:** Mix domain-specific data with a portion of general-purpose data during fine-tuning (e.g., 80% domain + 20% general instruction data).

4. **Regularization:**
   - **L2 regularization** (weight decay) penalizes large deviations from pre-trained weights.
   - **Elastic Weight Consolidation (EWC):** Adds a penalty proportional to each parameter's importance for previous tasks (estimated via Fisher information).

5. **Early stopping:** Monitor performance on both domain-specific and general benchmarks. Stop training when general capabilities begin to degrade.

6. **Multi-task learning:** Train on multiple tasks simultaneously rather than sequentially.

7. **Progressive/gradual unfreezing:** Start by training only the top layers, then progressively unfreeze deeper layers.

---

### 11. When should you choose fine-tuning over RAG over prompt engineering?

| Criteria | Prompt Engineering | RAG | Fine-Tuning |
|---|---|---|---|
| **Best for** | Quick iteration, general tasks, prototyping | Dynamic/frequently changing knowledge, factual grounding | Style/behavior change, domain-specific patterns, latency optimization |
| **Data requirements** | None (just examples in prompt) | A knowledge base / document corpus | Hundreds to thousands of labeled examples |
| **Knowledge freshness** | Static (whatever's in the prompt) | Real-time (retrieval at query time) | Static (baked into weights at training time) |
| **Setup cost** | Minutes | Days (indexing, retrieval pipeline, chunking) | Days to weeks (data curation, training, evaluation) |
| **Iteration speed** | Fastest | Medium | Slowest |
| **Inference cost** | Can be high (long prompts) | Medium (retrieval + generation) | Lower (shorter prompts, internalized behavior) |
| **Hallucination control** | Limited | Better (grounded in retrieved docs) | Limited (can still hallucinate) |
| **Use case examples** | "Summarize this text", "Translate to French" | "Answer questions about our 10K filing", "Search company docs" | "Always respond in our brand voice", "Output structured medical codes" |

**Decision framework:**
1. **Start with prompt engineering.** If it works, stop.
2. **Add RAG** if the model needs access to external, dynamic, or proprietary knowledge it wasn't trained on.
3. **Fine-tune** if you need consistent behavioral/stylistic changes, domain-specific reasoning patterns, or reduced inference cost that prompt engineering can't achieve.
4. **Combine**: Fine-tune + RAG is common — fine-tune for style/format, RAG for grounding in current facts.

---

### 12. How do you evaluate a fine-tuned model's performance?

**Automated metrics:**
- **Loss curves**: Training and validation loss should both decrease; divergence signals overfitting.
- **Perplexity**: Lower is better; measures how well the model predicts held-out text.
- **Task-specific metrics**: BLEU/ROUGE (text generation), F1/accuracy (classification), exact match (extraction), pass@k (code generation).
- **General benchmarks**: Run MMLU, HellaSwag, ARC, TruthfulQA, HumanEval to check for regression in general capabilities.

**Human evaluation:**
- **Side-by-side comparison**: Show outputs from the base model and fine-tuned model to human evaluators (blind). Measure win rate.
- **Likert scale ratings**: Rate outputs on helpfulness, accuracy, coherence, safety.
- **Domain expert review**: For specialized domains, only domain experts can assess correctness.

**LLM-as-judge:**
- Use a strong model (GPT-4, Claude) to compare outputs pairwise or rate them on defined rubrics. Correlates well with human judgments for many tasks but should be validated.

**Safety and regression testing:**
- Test for catastrophic forgetting: evaluate on general benchmarks before and after fine-tuning.
- Test for memorization: check if the model reproduces training examples verbatim.
- Red-team for safety regressions: ensure fine-tuning hasn't broken safety guardrails.

**A/B testing in production:**
- Deploy both models behind a feature flag, split traffic, and measure real-world metrics (user satisfaction, task completion, engagement).

---

### 13. What is synthetic data generation, and how do you use it for fine-tuning?

**Synthetic data generation** uses a capable LLM to create training examples rather than relying solely on human-written data.

**Methods:**

1. **Self-Instruct / Evol-Instruct:**
   - Start with a seed set of instructions.
   - Use an LLM to generate new instructions and corresponding responses.
   - Evol-Instruct (used for WizardLM) iteratively evolves instructions to increase complexity and diversity.

2. **Distillation-based generation:**
   - Use a strong teacher model (GPT-4, Claude) to generate responses for a set of prompts.
   - Train a smaller student model on these (prompt, teacher_response) pairs.

3. **Seed-and-expand:**
   - Write 10–50 gold-standard examples manually.
   - Prompt an LLM to generate variations while maintaining quality.

4. **Backtranslation / paraphrasing:**
   - Take existing data and use LLMs to rephrase inputs while keeping outputs consistent, increasing data diversity.

**Quality control for synthetic data:**
- Filter by a reward model or strong LLM judge.
- Human spot-check a random sample (10–20%).
- Remove duplicates and near-duplicates.
- Verify factual accuracy for knowledge-dependent tasks.
- Use rejection sampling: generate N responses, keep only the best-scored ones.

**Legal considerations:** Check the terms of service of the teacher model. Some (e.g., older OpenAI terms) restrict using outputs to train competing models. Anthropic's and newer OpenAI terms are more permissive for API-generated outputs.

---

### 14. What are the key hyperparameters for fine-tuning (learning rate, epochs, batch size, LoRA rank)?

| Hyperparameter | Typical Range | Notes |
|---|---|---|
| **Learning rate** | 1e-5 to 5e-5 (full FT), 1e-4 to 3e-4 (LoRA) | LoRA tolerates higher LR since base weights are frozen. Use cosine or linear decay schedule. |
| **Epochs** | 1–5 | Small datasets: 3–5 epochs. Large datasets: often 1 epoch suffices. Watch for overfitting. |
| **Batch size** | 4–128 (effective, with gradient accumulation) | Larger batches → smoother gradients but more memory. Use gradient accumulation to simulate large batches. |
| **LoRA rank (r)** | 4–64 | Higher rank = more expressiveness but more parameters. r=8–16 is a good default. Complex tasks may benefit from r=32–64. |
| **LoRA alpha** | Usually 2×rank (e.g., r=16, α=32) | Scaling factor. α/r is the effective scaling. Higher α → larger adapter contribution. |
| **LoRA target modules** | q_proj, v_proj (minimum); add k_proj, o_proj, gate_proj, up_proj, down_proj for more capacity | Targeting more modules increases capacity but also trainable params. |
| **Warmup ratio** | 3–10% of total steps | Prevents early instability. |
| **Weight decay** | 0.0–0.1 | Light regularization. |
| **Max sequence length** | Model-dependent | Truncate or filter data beyond this. Longer sequences = more memory. |
| **Gradient clipping** | 1.0 | Standard safeguard against gradient explosions. |

---

### 15. How do you fine-tune a model for a specific domain (legal, medical, finance)?

**Step 1 — Continual pre-training (optional but recommended):**
- Further pre-train on a large unlabeled domain corpus (legal briefs, PubMed abstracts, SEC filings) using the standard language modeling objective.
- This teaches the model domain vocabulary, jargon, and reasoning patterns.

**Step 2 — Curate domain-specific instruction data:**
- Collect or generate (instruction, response) pairs relevant to the domain.
- Examples: "Summarize this contract clause", "Extract adverse events from this clinical note", "Classify this transaction as suspicious or normal."
- Ensure data covers the full range of tasks the model will perform.

**Step 3 — Fine-tune with PEFT:**
- Use LoRA/QLoRA to preserve general capabilities while adding domain expertise.
- Mix in some general instruction data (10–20%) to prevent forgetting.

**Step 4 — Domain-specific evaluation:**
- Use domain benchmarks (e.g., LegalBench, PubMedQA, FinBen).
- Have domain experts evaluate a sample of outputs for accuracy.
- Test on real-world scenarios from the target deployment.

**Step 5 — Compliance and safety:**
- Legal: Ensure the model doesn't provide legal advice without disclaimers.
- Medical: Validate against clinical guidelines; ensure appropriate uncertainty.
- Finance: Test for compliance with regulations; check for bias in risk assessments.

**Step 6 — Combine with RAG:**
- Fine-tune for style/format, use RAG for grounding in current regulations, case law, or clinical guidelines that change over time.

---

### 16. What is continual pre-training, and when would you use it?

**Continual pre-training** (also called domain-adaptive pre-training) is the process of continuing the next-token prediction (language modeling) objective on a large, unlabeled, domain-specific corpus, after initial pre-training but before instruction fine-tuning.

**When to use it:**
- The target domain has highly specialized vocabulary, conventions, or reasoning patterns not well-represented in the general pre-training corpus (e.g., biomedical literature, legal case law, codebases in niche languages).
- You have a large volume of unlabeled domain text (millions of tokens).
- Instruction fine-tuning alone doesn't close the performance gap because the model lacks foundational domain understanding.

**How it fits in the training pipeline:**
```
General pre-training → Continual pre-training (domain corpus) → SFT (instructions) → RLHF/DPO (alignment)
```

**Key considerations:**
- Use a lower learning rate than original pre-training (typically 1e-5 to 5e-5).
- Risk of catastrophic forgetting is real — mix in some general text or use a replay buffer.
- Requires significant compute (more than SFT, less than pre-training from scratch).
- The result is a domain-adapted base model that you then instruction-tune.

---

### 17. How do you merge multiple LoRA adapters?

**Methods for merging LoRA adapters:**

1. **Simple weight merging (into base model):**
   ```
   W_merged = W_base + B₁A₁ + B₂A₂
   ```
   - Add each adapter's low-rank product directly to the base weights.
   - Risk: adapters trained independently may interfere (conflicting weight updates).

2. **Linear interpolation / weighted average:**
   ```
   W_merged = W_base + α₁(B₁A₁) + α₂(B₂A₂)
   ```
   - Weight each adapter's contribution with scaling factors α.
   - Tune α values empirically on a validation set.

3. **TIES Merging (Trim, Elect Sign, Merge):**
   - Trims small-magnitude deltas (noise).
   - Resolves sign conflicts by majority vote.
   - Averages the remaining values.
   - Better handles interference between adapters.

4. **DARE (Drop And Rescale):**
   - Randomly drops a fraction of adapter delta parameters.
   - Rescales the remaining ones to preserve expected magnitude.
   - Merged result is often better than naive averaging.

5. **Task Arithmetic:**
   - Compute task vectors: τ = W_finetuned - W_base for each adapter.
   - Merge: W_merged = W_base + Σ(λᵢ · τᵢ).
   - Can also negate task vectors to remove capabilities.

6. **Sequential application (stacking):**
   - Apply one adapter, merge into weights, then apply the next.
   - Order-dependent; experiment with different orderings.

**Best practice:** Evaluate merged models on all constituent tasks to ensure no single task regresses significantly.

---

### 18. What is the difference between SFT (Supervised Fine-Tuning) and alignment training?

| Aspect | SFT | Alignment Training |
|---|---|---|
| **Objective** | Learn to follow instructions and produce correct outputs | Learn to produce outputs that align with human values and preferences |
| **Training signal** | Direct supervision — (input, desired_output) pairs | Preference signal — "output A is better than output B" |
| **Loss function** | Cross-entropy (next-token prediction on target outputs) | Reward-based (PPO, DPO) or preference-based ranking loss |
| **Data type** | Demonstrations of correct behavior | Human preference comparisons, rankings |
| **What it teaches** | "What to say" | "What humans prefer" and "what not to say" |
| **Typical stage** | Comes first (after pre-training) | Comes after SFT |

**SFT** teaches the model to generate helpful responses by imitating expert demonstrations. It's supervised learning with explicit targets.

**Alignment training** goes further by teaching the model to distinguish between good and bad responses according to human values. It handles the fact that many possible responses are "correct" but some are better than others (more helpful, more honest, safer). Methods include RLHF (PPO), DPO (Direct Preference Optimization), and RLAIF.

**They are complementary:** SFT creates a capable instruction-following model; alignment training refines it to be more helpful, harmless, and honest.

---

### 19. What is RLAIF (RL from AI Feedback), and how does it differ from RLHF?

**RLAIF** replaces human annotators with an AI model (typically a strong LLM) for generating preference labels.

**Process:**
1. Generate multiple responses for each prompt.
2. An AI evaluator (e.g., Claude, GPT-4) ranks or scores the responses based on predefined criteria (helpfulness, harmlessness, accuracy).
3. Train a reward model on these AI-generated preferences.
4. Optimize the policy using RL (PPO) against the reward model.

**Alternatively:** Constitutional AI (Anthropic's approach) uses RLAIF where the AI critiques and revises its own outputs according to a set of principles (a "constitution").

| Aspect | RLHF | RLAIF |
|---|---|---|
| **Annotators** | Humans | AI model |
| **Cost** | Expensive (human labor) | Much cheaper |
| **Scale** | Limited by annotator availability | Essentially unlimited |
| **Consistency** | Variable (annotator disagreement) | More consistent (same model, same criteria) |
| **Bias** | Human biases | AI model's biases (potentially different biases) |
| **Quality ceiling** | Can capture nuanced human values | Limited by the evaluator model's judgment |
| **Bootstrapping problem** | None | The AI evaluator itself needed alignment |

**In practice:** Many teams use a hybrid — RLAIF for the bulk of preference data, with human annotations for high-stakes or ambiguous cases. Constitutional AI (RLAIF) has been shown to produce models competitive with RLHF while being significantly cheaper.

---

### 20. What is knowledge distillation for fine-tuning, and what are the legal considerations?

**Knowledge distillation** transfers knowledge from a large "teacher" model to a smaller "student" model.

**Methods for LLMs:**

1. **Response-based distillation (most common):**
   - Use the teacher model to generate responses for a set of prompts.
   - Fine-tune the student model on (prompt, teacher_response) pairs.
   - The student learns to mimic the teacher's behavior.

2. **Logit-based distillation:**
   - Train the student to match the teacher's output probability distribution (soft targets) using KL divergence loss:
     ```
     Loss = α · KL(softmax(z_t/T) || softmax(z_s/T)) + (1-α) · CE(y, z_s)
     ```
   - Temperature T > 1 softens the distribution, revealing more information about the teacher's learned structure.
   - Requires access to teacher logits (not always available via API).

3. **Feature-based distillation:**
   - Match intermediate representations (hidden states) between teacher and student.
   - Requires architectural compatibility or projection layers.

**Legal considerations:**
- **Terms of Service**: Some model providers explicitly prohibit using their API outputs to train competing models. Always check the ToS of the teacher model.
- **Licensing**: Open-source model licenses (Llama, Mistral) have varying restrictions on derivative models and commercial use.
- **Copyright**: Generated outputs may not be copyrightable, but the training data the teacher was trained on may be copyrighted. This is an evolving legal area.
- **Output ownership**: Clarify who owns the fine-tuned model and its outputs.
- **Data privacy**: If the teacher model was trained on or exposed to proprietary data, distillation may inadvertently leak that data.

---

### 21. Your fine-tuned LLM produces factually wrong outputs due to training data quality issues. How do you fix it?

**Diagnosis:**
- Identify specific categories of errors (factual inaccuracies, outdated info, made-up entities).
- Trace errors back to training data — find the offending examples.

**Fixes:**

1. **Data audit and cleaning:**
   - Manually review a sample of training data for factual accuracy.
   - Use automated fact-checking (cross-reference claims against trusted sources or a strong LLM).
   - Remove or correct inaccurate examples.

2. **Data filtering pipeline:**
   - Build a quality classifier: train a model to score data quality or use an LLM-as-judge to flag low-quality examples.
   - Filter by confidence scores — keep only examples above a quality threshold.

3. **Source verification:**
   - Weight training data by source reliability.
   - Prioritize expert-written or verified examples over scraped/synthetic data.

4. **Retrain with cleaned data:**
   - Fine-tune again from the base model (not from the corrupted checkpoint) using the cleaned dataset.
   - If using LoRA, simply train a new adapter.

5. **Add RAG for factual grounding:**
   - If the model needs to state facts, combine fine-tuning with RAG so facts come from a verified retrieval source, not from memorized (potentially wrong) weights.

6. **Post-processing:**
   - Add a fact-checking layer that validates model outputs against a knowledge base before serving to users.

---

### 22. You must choose between LoRA and full fine-tuning for a domain-specific assistant. How do you decide?

**Choose LoRA when:**
- You have limited GPU resources (LoRA uses ~10-30% of the memory of full FT).
- Your dataset is relatively small (< 10K examples) — LoRA's regularization effect (frozen base weights) reduces overfitting.
- You need to serve multiple domains — swap LoRA adapters (a few MBs each) on a single base model.
- General capabilities must be preserved — frozen base weights inherently prevent catastrophic forgetting.
- Fast iteration is important — LoRA trains faster and produces tiny checkpoints.

**Choose full fine-tuning when:**
- You have abundant compute and large datasets (100K+ examples).
- The domain is extremely different from the pre-training distribution (e.g., a new language, highly technical domain), and low-rank updates may not have enough capacity.
- Maximum performance is critical, and you've verified that LoRA leaves a meaningful performance gap.
- You're training a foundation model for your organization that won't need to serve other tasks.

**Decision process:**
1. Start with LoRA (r=16). Evaluate.
2. If performance is insufficient, increase rank (r=64) and target more modules.
3. If still insufficient, try QLoRA with a larger model first (a bigger model with LoRA often beats a smaller model with full fine-tuning).
4. Only resort to full fine-tuning if LoRA on the largest feasible model still falls short.

---

### 23. Your fine-tuned model memorized training data verbatim instead of learning patterns. How do you fix overfitting?

**Symptoms:** Model reproduces exact training examples, poor performance on held-out data, training loss much lower than validation loss.

**Fixes:**

1. **Reduce training duration:** Fewer epochs (often 1–3 is sufficient). Use early stopping based on validation loss.

2. **Increase data diversity:**
   - Add more training examples (even synthetic ones).
   - Augment existing data with paraphrases, reformulations.
   - Deduplicate the dataset — repeated examples accelerate memorization.

3. **Regularization:**
   - Increase weight decay (0.01–0.1).
   - Increase dropout (if configurable in the model).
   - For LoRA, reduce rank — lower rank constrains the adapter's capacity.

4. **Lower learning rate:** A high learning rate causes the model to quickly overfit to the training distribution.

5. **Increase batch size:** Larger batches produce smoother gradient estimates, reducing overfitting to individual examples.

6. **Data quality over quantity:** Remove noisy, contradictory, or redundant examples that encourage memorization.

7. **Evaluation during training:** Monitor validation metrics at every epoch/N steps. Plot train vs. val loss to catch the divergence point.

8. **Label smoothing:** Soften the target distribution (e.g., 0.9 for correct token, 0.1 distributed across others) to prevent the model from becoming overconfident.

---

### 24. Your fine-tuned LLM forgot its general capabilities after domain-specific fine-tuning. How do you fix catastrophic forgetting?

**Immediate fix:**
- Go back to the base model and re-fine-tune with a better strategy (below).

**Prevention strategies for the re-do:**

1. **Switch to LoRA/QLoRA:** Freeze base weights entirely. This is the single most effective fix.

2. **Data mixing:** Blend domain data with general instruction data:
   - 70–80% domain-specific + 20–30% general (e.g., from Open-Orca, Alpaca, Dolly).
   - This explicitly trains the model to maintain general capabilities.

3. **Lower learning rate:** Use 1e-5 or lower for full fine-tuning to minimize weight disruption.

4. **Fewer epochs:** Often 1–2 epochs is sufficient. Monitor general benchmarks (MMLU, HellaSwag) during training and stop when they start dropping.

5. **Progressive unfreezing:** First train only top layers, then gradually unfreeze deeper layers. This preserves foundational representations in lower layers.

6. **EWC or similar regularization:** Penalize changes to parameters that are important for general tasks (estimated via Fisher information matrix).

7. **Multi-task training:** Instead of sequential fine-tuning, train on domain tasks and general tasks simultaneously in each batch.

8. **Checkpoint selection:** Don't always pick the checkpoint with the lowest domain-specific loss. Pick the one with the best trade-off between domain performance and general capability retention.

---

### 25. Your RLHF preference data has low annotator agreement. How do you ensure data quality?

**Diagnosis:**
- Compute inter-annotator agreement (Cohen's Kappa, Fleiss' Kappa, or Krippendorff's alpha).
- Identify which types of comparisons have the lowest agreement (ambiguous prompts? subjective preferences? close-quality responses?).

**Improvement strategies:**

1. **Refine annotation guidelines:**
   - Make criteria more specific and objective (e.g., instead of "which is better?", define: "which is more factually accurate, complete, and clearly written?").
   - Provide concrete examples of edge cases with expected judgments.
   - Create rubrics with separate dimensions (helpfulness, accuracy, safety, coherence) rather than a single holistic rating.

2. **Annotator calibration:**
   - Run calibration sessions where annotators discuss disagreements.
   - Use gold-standard examples with known correct labels to detect and correct annotator drift.
   - Remove or retrain consistently disagreeing annotators.

3. **Increase annotations per example:**
   - Collect 3–5 annotations per comparison and use majority vote.
   - Weight annotators by their agreement with expert consensus.

4. **Filter ambiguous comparisons:**
   - Remove pairs where the margin of preference is very small (annotators genuinely can't tell which is better).
   - Focus the reward model on clear-cut preferences.

5. **Use a soft label approach:**
   - Instead of binary (A > B), use the distribution of votes (e.g., 70% prefer A, 30% prefer B) as a soft target in reward model training.

6. **Supplement with RLAIF:**
   - Use AI-generated preferences for clear-cut cases, reserve human annotation for ambiguous or high-stakes examples.

---

## Part 2: Vector Databases and Embeddings

---

### 26. What are embeddings in the context of AI engineering?

**Embeddings** are dense, fixed-dimensional vector representations of data (text, images, audio, etc.) in a continuous vector space where semantic similarity is captured by geometric proximity.

**Key properties:**
- **Dimensionality**: Typically 256–3072 dimensions depending on the model.
- **Semantic meaning**: Similar concepts map to nearby vectors. "King" and "Queen" are closer together than "King" and "Banana."
- **Learned representations**: Embeddings are produced by neural networks trained on large datasets, capturing statistical patterns of co-occurrence and meaning.

**How they're used in AI engineering:**
- **Semantic search / RAG**: Convert documents and queries to embeddings, find documents whose embeddings are closest to the query embedding.
- **Clustering**: Group similar documents, users, or products.
- **Classification**: Use embeddings as features for downstream classifiers.
- **Anomaly detection**: Identify outliers in embedding space.
- **Recommendation systems**: Find items with similar embeddings to a user's preferences.

**Common embedding models:** OpenAI `text-embedding-3-small/large`, Cohere `embed-v3`, sentence-transformers (`all-MiniLM-L6-v2`, `all-mpnet-base-v2`), BGE, E5, GTE, Nomic.

---

### 27. How do embedding models convert text to vectors?

**Architecture (typical modern embedding model):**

1. **Tokenization**: Input text is split into subword tokens using the model's tokenizer (BPE, WordPiece, or SentencePiece).

2. **Token embedding lookup**: Each token is mapped to a learned embedding vector from a vocabulary table.

3. **Positional encoding**: Position information is added (sinusoidal, learned, or RoPE) so the model understands token order.

4. **Transformer encoding**: Tokens pass through multiple transformer encoder layers (self-attention + FFN), producing contextualized representations for each token. Each token's representation is influenced by all other tokens.

5. **Pooling**: The per-token representations are aggregated into a single fixed-length vector:
   - **[CLS] token**: Use the representation of a special classification token (BERT-style).
   - **Mean pooling**: Average all token representations (most common for sentence-transformers).
   - **Max pooling**: Take the element-wise max across token representations.
   - **Last token**: Use the final token's representation (GPT-style decoder models adapted for embeddings).

6. **Normalization**: The resulting vector is often L2-normalized to unit length so that cosine similarity equals dot product.

**Training objectives for embedding models:**
- **Contrastive learning**: Bring embeddings of semantically similar pairs closer and push dissimilar pairs apart (InfoNCE, triplet loss).
- **MNRL (Multiple Negatives Ranking Loss)**: Given (query, positive_doc), use other batch items as in-batch negatives.
- **Matryoshka Representation Learning**: Train embeddings to be useful at multiple truncated dimensions.

---

### 28. What is the difference between sparse and dense embeddings?

| Aspect | Sparse Embeddings | Dense Embeddings |
|---|---|---|
| **Dimensionality** | Very high (vocabulary size, ~30K–100K+) | Low to moderate (256–3072) |
| **Values** | Mostly zeros, few non-zero entries | All dimensions have non-zero values |
| **What they capture** | Lexical/keyword overlap | Semantic meaning |
| **Example methods** | TF-IDF, BM25, SPLADE, learned sparse | BERT, Sentence-Transformers, OpenAI embeddings |
| **Strengths** | Exact keyword matching, interpretable, fast (inverted index), no training needed (BM25) | Captures synonyms, paraphrases, conceptual similarity |
| **Weaknesses** | Misses synonyms/paraphrases ("car" ≠ "automobile") | Can miss exact keywords, less interpretable, requires embedding model |
| **Storage** | Efficient (only store non-zero entries) | Every dimension stored |
| **Search method** | Inverted index (exact, fast) | ANN algorithms (HNSW, IVF, etc.) |

**Sparse example (BM25-style):**
"The cat sat on the mat" → `{cat: 0.8, sat: 0.6, mat: 0.7, ...}` (most dimensions = 0)

**Dense example:**
"The cat sat on the mat" → `[0.12, -0.34, 0.56, ..., 0.78]` (all 768 dimensions populated)

**Best practice:** Hybrid search combines both — use sparse for keyword recall and dense for semantic understanding, then fuse the scores (Reciprocal Rank Fusion, linear combination).

---

### 29. Explain cosine similarity, dot product, and Euclidean distance for vector search.

**Cosine Similarity:**
```
cos(A, B) = (A · B) / (||A|| × ||B||)
```
- Range: [-1, 1] (1 = identical direction, 0 = orthogonal, -1 = opposite).
- **Ignores magnitude**, only considers angle/direction.
- Best when you care about semantic orientation, not vector length.
- If vectors are L2-normalized (unit vectors), cosine similarity = dot product.

**Dot Product (Inner Product):**
```
A · B = Σ(aᵢ × bᵢ)
```
- Range: unbounded.
- **Sensitive to both direction and magnitude**.
- Higher magnitude vectors get higher scores even if direction is the same.
- Useful when magnitude carries information (e.g., importance, confidence).
- With normalized vectors: equivalent to cosine similarity.

**Euclidean Distance (L2):**
```
d(A, B) = √(Σ(aᵢ - bᵢ)²)
```
- Range: [0, ∞) — lower is more similar.
- Measures the straight-line distance between points.
- Sensitive to both direction and magnitude.
- Relation to cosine: for normalized vectors, Euclidean distance and cosine similarity are monotonically related: `d² = 2(1 - cos(A,B))`.

**When to use which:**
- **Cosine similarity**: Default choice for text embeddings. Most embedding models are designed for it.
- **Dot product**: When vectors are already normalized (equivalent to cosine) or when magnitude is meaningful (e.g., Maximum Inner Product Search / MIPS).
- **Euclidean distance**: When absolute position in vector space matters; less common for text but used in some clustering scenarios.

---

### 30. What is a vector database, and how does it differ from a traditional database?

**A vector database** is a specialized database designed to store, index, and efficiently retrieve high-dimensional vectors using similarity search (nearest neighbor queries).

| Aspect | Traditional Database (SQL/NoSQL) | Vector Database |
|---|---|---|
| **Primary query type** | Exact match, range, joins (WHERE, GROUP BY) | Similarity search (find K nearest neighbors) |
| **Data model** | Rows/columns, documents, key-value | Vectors + metadata |
| **Indexing** | B-trees, hash indexes, full-text indexes | ANN indexes (HNSW, IVF, PQ, ScaNN) |
| **Query result** | Exact matches | Approximate matches ranked by similarity |
| **Use case** | CRUD operations, transactions, analytics | Semantic search, RAG, recommendations, anomaly detection |

**Core capabilities of vector databases:**
- **ANN (Approximate Nearest Neighbor) search**: Trade small accuracy loss for massive speed gains over brute-force search.
- **Index types**: HNSW (graph-based, high recall, memory-heavy), IVF (partition-based, good for large datasets), PQ (Product Quantization, compressed vectors).
- **Metadata filtering**: Filter by metadata attributes before or after vector search (e.g., "find similar documents from the last 30 days").
- **CRUD operations**: Insert, update, delete vectors with real-time index updates.

**Popular vector databases:** Pinecone, Weaviate, Milvus, Qdrant, Chroma, pgvector (PostgreSQL extension), FAISS (library, not a database).

---

### 31. How do you choose the right embedding model for your use case?

**Evaluation criteria:**

1. **Quality / benchmark performance:**
   - Check MTEB (Massive Text Embedding Benchmark) leaderboard for your task type (retrieval, classification, clustering, semantic similarity).
   - Evaluate on your own data — benchmark rankings don't always transfer.

2. **Dimensionality:**
   - Higher dimensions (1024–3072) → better quality but more storage and slower search.
   - Lower dimensions (256–384) → cheaper, faster, and often sufficient.
   - Matryoshka models (e.g., OpenAI `text-embedding-3-*`) let you truncate dimensions flexibly.

3. **Max sequence length:**
   - Most models handle 512 tokens. Some handle 8K+ (e.g., Jina, Nomic).
   - If your documents are long, you need a long-context embedding model or a chunking strategy.

4. **Language support:**
   - Multilingual models (e.g., Cohere `embed-multilingual`, multilingual-e5) for non-English or mixed-language corpora.

5. **Cost:**
   - API-based (OpenAI, Cohere, Voyage): pay per token. Simple to deploy.
   - Self-hosted (sentence-transformers, BGE, GTE): free inference but need GPU infrastructure.

6. **Latency:**
   - Smaller models (MiniLM: 22M params) are faster than large models (E5-large: 335M params).
   - Consider whether you're embedding at ingest time (batch, latency less critical) or query time (real-time, latency matters).

7. **Domain specificity:**
   - General models work well for most tasks.
   - For highly specialized domains, consider fine-tuning the embedding model on domain data.

**Decision flow:**
1. Start with a strong general model (OpenAI `text-embedding-3-small`, `all-MiniLM-L6-v2`, or `bge-small-en-v1.5`).
2. Evaluate retrieval quality on your data.
3. If quality is insufficient, try a larger model.
4. If still insufficient, fine-tune the embedding model on your domain data.

---

### 32. What is embedding dimensionality, and how does it affect performance and cost?

**Embedding dimensionality** is the number of components in the vector representation (e.g., 384, 768, 1024, 1536, 3072).

**Impact on performance:**
- **Higher dimensions** can encode more nuanced semantic distinctions — more "axes" to represent meaning.
- **Diminishing returns**: Going from 384 → 768 often gives a meaningful quality boost. Going from 1536 → 3072 gives a much smaller one.
- **Too high**: Can lead to the "curse of dimensionality" where distances between all points become similar, making nearest-neighbor search less discriminative.

**Impact on cost and operations:**

| Aspect | Low-dim (256–384) | High-dim (1024–3072) |
|---|---|---|
| **Storage per vector** | 1–1.5 KB (float32) | 4–12 KB (float32) |
| **1M vectors storage** | ~1–1.5 GB | ~4–12 GB |
| **Search latency** | Faster (fewer computations) | Slower |
| **Index memory** | Lower (HNSW graph is smaller) | Higher |
| **Embedding computation** | Faster (smaller model) | Slower (larger model) |

**Matryoshka Representation Learning (MRL):**
- Models trained with MRL (e.g., OpenAI `text-embedding-3-*`, Nomic v1.5) produce embeddings where the first N dimensions are a valid lower-dimensional embedding.
- You can truncate from 3072 → 1024 → 256 with graceful quality degradation.
- This allows choosing the quality/cost tradeoff at deployment time without retraining.

---

### 33. How do you handle embedding drift when the embedding model is updated?

**Embedding drift** occurs when you update your embedding model and the new model produces vectors in a different vector space than the old model. Old and new embeddings are incompatible — you can't compare them directly.

**Strategies:**

1. **Full re-embedding (gold standard):**
   - Re-embed your entire corpus with the new model.
   - Pros: Clean, correct, no compatibility issues.
   - Cons: Expensive for large corpora (billions of documents), causes downtime or requires parallel infrastructure.

2. **Shadow deployment / blue-green:**
   - Build a new index with the new model alongside the old one.
   - Run both in parallel, gradually shift traffic.
   - Validate quality before decommissioning the old index.

3. **Versioned indexes:**
   - Maintain separate vector indexes per embedding model version.
   - Route queries to the appropriate index based on when documents were embedded.
   - Gradually backfill older documents with new embeddings.

4. **Embedding translation / alignment:**
   - Train a lightweight linear mapping (Procrustes alignment) from old embedding space to new embedding space.
   - Pros: Cheap, no need to re-embed.
   - Cons: Approximation — some quality loss, especially if the spaces are very different.

5. **Incremental re-embedding:**
   - Re-embed documents in priority order (most-accessed first).
   - Use a queue to gradually re-embed the long tail.

**Best practice:** Design your system to handle re-embedding from the start — store raw text alongside vectors so you can re-embed when needed. Version your embedding model and track which model version produced each vector.

---

### 34. What are multi-modal embeddings, and how are they generated?

**Multi-modal embeddings** map data from different modalities (text, images, audio, video) into a shared vector space where semantically similar items are close regardless of modality.

**How they're generated:**

1. **Contrastive Learning (CLIP-style):**
   - Train separate encoders for each modality (e.g., a Vision Transformer for images, a text transformer for text).
   - Use contrastive loss (InfoNCE) on paired data (image-caption pairs) to align the two embedding spaces:
     - Matching (image, caption) pairs are pulled together.
     - Non-matching pairs are pushed apart.
   - Result: Text and images live in the same vector space. You can search images with text queries and vice versa.

2. **Late fusion:**
   - Encode each modality separately, then combine (concatenate, average, or use a learned fusion layer).

3. **Cross-attention / early fusion:**
   - Feed multiple modalities into a single transformer with cross-attention between modalities.
   - Examples: Flamingo, GPT-4V (internal representations).

**Popular multi-modal embedding models:**
- **CLIP** (OpenAI): Text + Image.
- **SigLIP**: Improved CLIP variant.
- **ImageBind** (Meta): Binds 6 modalities (text, image, audio, depth, thermal, IMU).
- **CLAP**: Text + Audio.

**Use cases:**
- Cross-modal search ("find images matching this text description").
- Multi-modal RAG (retrieve relevant images and text for a query).
- Content moderation (match text descriptions to visual content).

---

### 35. How do you index and query multi-tenant data in a vector database?

**Multi-tenancy** means multiple users, organizations, or customers share the same vector database infrastructure, but each tenant's data must be isolated.

**Approaches:**

1. **Metadata-based filtering (most common):**
   - Store a `tenant_id` field as metadata on every vector.
   - At query time, apply a metadata filter: `filter={"tenant_id": "customer_123"}`.
   - Pros: Simple, single index, efficient storage.
   - Cons: Filter performance can degrade with many tenants; the ANN index itself isn't tenant-aware, so it may scan many irrelevant vectors before finding enough matches.

2. **Namespace/partition isolation:**
   - Some vector databases (Pinecone, Qdrant) support namespaces or collections per tenant.
   - Each tenant gets a separate partition within the same infrastructure.
   - Pros: True data isolation, no cross-tenant data leakage, better query performance.
   - Cons: More partitions = more index overhead; thousands of tenants can be expensive.

3. **Separate indexes/collections per tenant:**
   - Spin up a dedicated vector index for each tenant.
   - Pros: Complete isolation, independent scaling.
   - Cons: Operational overhead, wasted resources for small tenants, doesn't scale to thousands of tenants.

4. **Hybrid approach:**
   - Large tenants get dedicated namespaces/collections.
   - Small tenants share a pooled index with metadata filtering.

**Security considerations:**
- Always enforce tenant isolation at the application layer — never trust client-side filtering.
- Use server-side middleware to inject `tenant_id` into every query.
- Audit access patterns to detect cross-tenant data leakage.

---

### 36. What is quantization of embeddings, and how does it reduce storage costs?

**Embedding quantization** reduces the precision of vector components to use fewer bits per dimension, significantly reducing storage and memory requirements.

**Types:**

1. **Scalar quantization (SQ):**
   - Reduce each float32 dimension from 32 bits to 8 bits (int8) or 16 bits (float16).
   - **int8**: 4x memory reduction. Each dimension is linearly mapped from its observed range to [0, 255].
   - **binary**: 32x reduction. Each dimension becomes 1 bit (positive → 1, negative → 0). Uses Hamming distance. Significant quality loss.

2. **Product Quantization (PQ):**
   - Split each vector into subvectors (segments).
   - For each segment, learn a codebook of representative centroids (via K-means).
   - Replace each subvector with its nearest centroid's index (e.g., 8-bit index = 256 centroids).
   - **Example**: 768-dim vector split into 96 subvectors of 8 dims each, each encoded as 1 byte → 96 bytes vs. 3072 bytes (32x reduction).

3. **Matryoshka + truncation:**
   - Not quantization per se, but reducing dimensionality from 3072 → 512 achieves 6x storage reduction while preserving quality (for MRL-trained models).

**Impact:**

| Method | Bits/dim | Memory (1M × 768-dim) | Quality Loss |
|---|---|---|---|
| float32 | 32 | ~3 GB | None (baseline) |
| float16 | 16 | ~1.5 GB | Negligible |
| int8 (SQ) | 8 | ~768 MB | Small (~1-2% recall drop) |
| PQ (96 subvectors) | ~1 | ~96 MB | Moderate (~3-5% recall drop) |
| Binary | 1 | ~96 MB | Large (use for pre-filtering only) |

**Common strategy:** Use full-precision or float16 for the primary index (accurate search), and PQ or binary for a coarse pre-filter (fast candidate generation), then re-rank candidates with full-precision vectors.

---

### 37. How do you benchmark and evaluate embedding model quality?

**Standard benchmarks:**

1. **MTEB (Massive Text Embedding Benchmark):**
   - 56+ datasets across 8 task categories: retrieval, classification, clustering, pair classification, reranking, STS (semantic textual similarity), summarization, and more.
   - The standard leaderboard for comparing embedding models.

2. **BEIR (Benchmarking IR):**
   - 18 retrieval datasets across diverse domains (bio, finance, StackOverflow, etc.).
   - Measures zero-shot retrieval quality.

**Custom evaluation (on your data):**

1. **Retrieval metrics:**
   - **Recall@K**: Of all relevant documents, what fraction appears in the top-K results?
   - **NDCG@K**: Normalized Discounted Cumulative Gain — considers ranking position.
   - **MRR**: Mean Reciprocal Rank — where does the first relevant result appear?
   - **Precision@K**: Of the top-K results, what fraction is relevant?

2. **Semantic similarity:**
   - Spearman correlation between embedding cosine similarity and human-annotated similarity scores on STS benchmarks.

3. **Downstream task performance:**
   - For RAG: measure end-to-end answer quality (LLM accuracy given retrieved context).
   - For classification: use embeddings as features, measure F1/accuracy.

4. **Latency and throughput:**
   - Embedding generation speed (tokens/sec or documents/sec).
   - Query latency at different index sizes.

5. **A/B testing in production:**
   - Compare user satisfaction, click-through rates, or task completion with different embedding models.

**Best practice:** Always evaluate on your own data distribution, not just public benchmarks. A model ranked #1 on MTEB may not be #1 for your specific domain.

---

### 38. What is the role of metadata in vector databases?

**Metadata** is structured information attached to each vector that provides context beyond the vector representation itself.

**Common metadata fields:**
- **Source information**: document ID, URL, file path, page number, chunk index.
- **Temporal**: created_at, updated_at, publication date.
- **Categorical**: category, department, language, author, document type.
- **Access control**: tenant_id, user_id, permission level.
- **Content attributes**: title, summary, word count, confidence score.

**How metadata is used:**

1. **Pre-filtering**: Narrow the search space before vector similarity search.
   - "Find similar documents from the last 30 days" → filter by date, then do vector search.

2. **Post-filtering**: Do vector search first, then filter results by metadata.
   - Can result in fewer than K results if many candidates are filtered out.

3. **Hybrid ranking**: Combine vector similarity scores with metadata-based signals (recency boost, authority score).

4. **Deduplication**: Use source metadata to avoid returning multiple chunks from the same document.

5. **Access control**: Enforce tenant isolation, permission-based retrieval.

6. **Debugging and observability**: Track which chunks are retrieved and from where, enabling retrieval quality analysis.

**Best practices:**
- Index metadata fields you'll filter on frequently (most vector DBs support metadata indexes).
- Keep metadata lightweight — don't store full document text as metadata.
- Use consistent schemas across your corpus.

---

### 39. How do you handle large-scale vector search with billions of vectors?

**Challenges at scale:**
- Memory: 1B vectors × 768 dims × 4 bytes = ~3 TB in float32.
- Latency: Brute-force is impossible; ANN quality can degrade.
- Index build time: HNSW on billions of vectors takes days.

**Strategies:**

1. **Approximate Nearest Neighbor (ANN) indexes:**
   - **HNSW (Hierarchical Navigable Small World):** Graph-based, high recall, but memory-resident. Best for up to hundreds of millions of vectors per node.
   - **IVF (Inverted File Index):** Partition vectors into clusters, search only relevant clusters. Supports disk-based search.
   - **IVF + PQ:** Combine partitioning with product quantization for compressed in-memory search.

2. **Quantization (reduce memory per vector):**
   - SQ8: 4x reduction.
   - PQ: 16–32x reduction.
   - Binary: 32x reduction (use as first-stage filter only).

3. **Sharding / distributed search:**
   - Shard the index across multiple machines.
   - Query all shards in parallel, merge results.
   - Managed services (Pinecone, Milvus, Qdrant Cloud) handle this automatically.

4. **Tiered storage:**
   - Hot tier: frequently accessed vectors in memory (HNSW).
   - Warm tier: less frequent vectors on SSD (DiskANN, IVF-PQ with disk-backed storage).
   - Cold tier: archive with on-demand loading.

5. **Pre-filtering to reduce search space:**
   - Use metadata filters to narrow candidates before vector search.
   - Partition indexes by date, category, or tenant.

6. **Dimensionality reduction:**
   - Use MRL-trained models and truncate to lower dimensions.
   - PCA (but retrain the projection for domain data).

7. **Multi-stage retrieval:**
   - Stage 1: Coarse search with binary/PQ vectors (fast, high recall).
   - Stage 2: Re-rank top candidates with full-precision vectors.
   - Stage 3: LLM-based reranking (Cohere Rerank, cross-encoder).

---

### 40. What is hybrid search (combining keyword search with vector search)?

**Hybrid search** combines lexical/keyword search (BM25, TF-IDF) with semantic/vector search to leverage the strengths of both approaches.

**Why:**
- **Vector search** excels at semantic understanding (synonyms, paraphrases, conceptual similarity) but can miss exact keyword matches.
- **Keyword search** excels at exact term matching, rare terms, proper nouns, and codes/IDs but misses semantic relationships.
- Combined, they achieve higher recall and relevance than either alone.

**Implementation approaches:**

1. **Reciprocal Rank Fusion (RRF):**
   ```
   RRF_score(doc) = Σ 1/(k + rank_i(doc))
   ```
   - k is a constant (typically 60).
   - Combine rankings from keyword and vector search.
   - Simple, effective, no tuning needed.

2. **Linear score combination:**
   ```
   final_score = α × vector_score + (1 - α) × keyword_score
   ```
   - Normalize scores to [0, 1] first.
   - Tune α on a validation set (typically 0.5–0.7 weight on vector).

3. **Native hybrid search:**
   - Some databases (Weaviate, Qdrant, Elasticsearch 8+, Vespa) support hybrid search natively with configurable fusion.

4. **Learned sparse + dense:**
   - Use SPLADE (learned sparse representations) alongside dense embeddings.
   - SPLADE learns which terms are important and expands queries with relevant terms.

**Architecture:**
```
Query → [Keyword Search (BM25)] → Top-K₁ results
      → [Vector Search (ANN)]  → Top-K₂ results
      → [Fusion (RRF/Linear)]  → Top-K final results
      → [Reranker (optional)]  → Final ranked results
```

---

### 41. How do you fine-tune an embedding model for a specific domain?

**Why fine-tune:**
- General embedding models may not capture domain-specific semantic relationships (e.g., medical terms, legal jargon, internal company terminology).
- Fine-tuning can significantly improve retrieval quality in specialized domains.

**Training data preparation:**

1. **Positive pairs (most important):**
   - (query, relevant_document) pairs from your domain.
   - Sources: search logs (query + clicked document), QA pairs, (title, body) pairs, (question, answer) pairs.
   - Minimum: 1,000 pairs, ideally 10K+.

2. **Hard negatives:**
   - Documents that are superficially similar but not relevant.
   - Mine hard negatives by retrieving with the current model and selecting non-relevant results.
   - Much more effective than random negatives.

**Fine-tuning process:**

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

model = SentenceTransformer('BAAI/bge-base-en-v1.5')

train_examples = [
    InputExample(texts=["query", "positive_doc"]),
    InputExample(texts=["query", "positive_doc", "hard_negative_doc"]),
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)

# Choose loss function
train_loss = losses.MultipleNegativesRankingLoss(model)
# Or with hard negatives:
# train_loss = losses.TripletLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path='fine-tuned-embeddings'
)
```

**Key considerations:**
- Use a small learning rate (1e-5 to 2e-5) to avoid destroying pre-trained knowledge.
- Evaluate on a held-out retrieval test set using Recall@K and NDCG@K.
- Compare against the base model to ensure fine-tuning actually improved quality.
- After fine-tuning, you must re-embed your entire corpus with the new model.

---

### 42. Your vector database for RAG is consuming too much memory. How do you reduce it?

**Diagnosis:**
- Profile memory usage: vectors vs. index structures vs. metadata.
- Calculate: `num_vectors × dimensions × bytes_per_dim + index_overhead`.

**Solutions (ordered by impact and complexity):**

1. **Quantize vectors:**
   - float32 → float16: 2x reduction, negligible quality loss.
   - float32 → int8: 4x reduction, minimal quality loss (~1-2% recall drop).
   - Product Quantization: 16–32x reduction, moderate quality loss.

2. **Reduce dimensionality:**
   - If using an MRL-trained model, truncate dimensions (e.g., 1536 → 512).
   - If not MRL-trained, apply PCA (retrain on your data distribution).

3. **Move to disk-based indexes:**
   - DiskANN: keeps graph structure in memory but vectors on SSD.
   - IVF-PQ with memory-mapped vectors.
   - Milvus and Qdrant support tiered storage.

4. **Reduce corpus size:**
   - Deduplicate documents (exact and near-duplicate removal).
   - Remove low-quality or irrelevant chunks.
   - Use smarter chunking to reduce the total number of chunks.

5. **Use a smaller embedding model:**
   - Switch from a 1536-dim model to a 384-dim model if quality is acceptable.

6. **Shard across machines:**
   - Distribute the index across multiple nodes, each holding a subset.

7. **Separate hot/cold storage:**
   - Keep frequently accessed vectors in memory.
   - Move rarely accessed vectors to disk or archive.

---

### 43. Your vector database cannot scale to millions of embeddings. How do you fix the bottleneck?

**Identify the bottleneck:**
- **Memory**: Index doesn't fit in RAM → quantization, disk-based indexing.
- **Query latency**: Searches are too slow → better ANN parameters, sharding.
- **Write throughput**: Insertions are slow → batch inserts, async indexing.
- **Index build time**: Rebuilding takes too long → incremental indexing.

**Solutions:**

1. **Switch to a scalable vector database:**
   - From FAISS (single-node) → Milvus, Qdrant, Weaviate, or Pinecone (distributed, auto-scaling).

2. **Horizontal sharding:**
   - Distribute vectors across multiple nodes by hash or partition key.
   - Query all shards in parallel, merge results.

3. **Better index configuration:**
   - **HNSW parameters**: Increase `ef_construction` for build quality, tune `ef_search` for query quality/speed tradeoff, adjust `M` (connections per node).
   - **IVF**: Increase `nlist` (number of partitions), tune `nprobe` (partitions searched).

4. **Quantization:**
   - Reduce per-vector memory with SQ8 or PQ, allowing more vectors to fit in RAM.

5. **Tiered indexing:**
   - Recent/popular vectors in HNSW (fast, in-memory).
   - Older vectors in IVF-PQ on disk (slower, but handles billions).

6. **Batch operations:**
   - Insert in batches of 100–1000 rather than one at a time.
   - Use async/background index building rather than blocking on each insert.

7. **Pre-filtering optimization:**
   - Partition by tenant, date range, or category to reduce the effective search space per query.

---

### 44. Your new embedding model has different dimensions from the existing vectors in production. How do you handle the mismatch?

**The problem:** Old model outputs 768-dim vectors, new model outputs 1024-dim. You can't compute similarity between vectors of different dimensions.

**Solutions:**

1. **Full re-embedding (recommended):**
   - Re-embed the entire corpus with the new model.
   - Use a blue-green deployment: build a new index alongside the old one, switch traffic once ready.
   - This is the cleanest solution but most expensive.

2. **Gradual migration with dual indexes:**
   - Run two indexes in parallel (old model index + new model index).
   - New documents go into the new index.
   - Background job gradually re-embeds old documents into the new index.
   - Query both indexes, merge results (weighted by recency or index coverage).
   - Decommission old index when migration is complete.

3. **Learned projection / alignment:**
   - Train a lightweight linear projection layer that maps 768-dim → 1024-dim (or vice versa).
   - Use paired data (same text embedded by both models) to learn the mapping via MSE or cosine loss.
   - Apply this projection to old vectors at query time or as a one-time batch transformation.
   - Pros: Fast, cheap.
   - Cons: Approximation — quality degradation, especially if the embedding spaces are very different.

4. **Zero-padding / truncation (last resort):**
   - Pad 768-dim vectors with zeros to 1024-dim, or truncate 1024-dim to 768.
   - Only viable if the new model supports Matryoshka representations and the truncation doesn't destroy quality.
   - Generally NOT recommended — geometrically incorrect.

**Best practice:** Store raw text alongside vectors. This makes re-embedding straightforward when models change. Plan for model upgrades in your architecture from day one.

---

### 45. Your vector search returns irrelevant results despite high similarity scores. How do you fix it?

**Common causes and fixes:**

1. **Embedding model mismatch:**
   - The embedding model doesn't understand your domain vocabulary.
   - **Fix**: Fine-tune the embedding model on domain-specific (query, relevant_doc) pairs.

2. **Poor chunking strategy:**
   - Chunks are too small (no context) or too large (diluted meaning).
   - **Fix**: Experiment with chunk sizes (256–1024 tokens), use overlapping chunks, try semantic chunking (split at topic boundaries).

3. **Query-document asymmetry:**
   - Queries are short and vague; documents are long and detailed.
   - **Fix**: Use query expansion (LLM rewrites the query), hypothetical document embeddings (HyDE — generate a hypothetical answer, embed that).

4. **Cosine similarity limitations:**
   - High cosine similarity doesn't guarantee relevance — vectors might be close in embedding space but semantically off for your task.
   - **Fix**: Add a reranker (cross-encoder) as a second stage. Cross-encoders jointly encode (query, document) and are much more accurate than bi-encoders for relevance scoring.

5. **Missing metadata filters:**
   - The search returns documents from irrelevant categories or time periods.
   - **Fix**: Add metadata filters (date range, category, source) to narrow the search space.

6. **Hybrid search:**
   - Pure vector search misses exact keyword matches.
   - **Fix**: Combine BM25 + vector search with RRF or linear fusion.

7. **Stale or incorrect embeddings:**
   - Document content was updated but embeddings weren't re-generated.
   - **Fix**: Implement an embedding update pipeline triggered by content changes.

8. **Retrieval evaluation:**
   - Build a test set of (query, relevant_documents) pairs and measure Recall@K and NDCG@K to quantify the problem and track improvements.

---

### 46. You deployed a new embedding model, and search quality crashed overnight. How do you handle embedding drift?

**Immediate actions:**

1. **Rollback:** Switch back to the old model and old index immediately. Restore search quality while you investigate.

2. **Root cause analysis:**
   - The new model produces vectors in a different embedding space.
   - Old documents are embedded with the old model; new queries are embedded with the new model → misalignment.
   - Alternatively, the new model may genuinely be worse for your domain.

**Recovery plan:**

1. **Validate the new model offline:**
   - Re-embed a test set of documents with the new model.
   - Embed test queries with the new model.
   - Evaluate Recall@K, NDCG@K on your evaluation set.
   - Compare against the old model. If the new model is actually worse, don't migrate.

2. **If the new model is better (on consistent embeddings):**
   - Re-embed the entire corpus with the new model before switching.
   - Use blue-green deployment: build the new index in the background, switch traffic atomically.

3. **Implement safeguards for future upgrades:**
   - Always validate new models on your evaluation set before deploying.
   - Never mix embeddings from different models in the same index.
   - Use canary deployments: route 5% of traffic to the new model/index first, monitor quality metrics.
   - Maintain the ability to rollback instantly (keep the old index alive for 24–48 hours after migration).

4. **Monitoring:**
   - Set up automated retrieval quality monitoring (track average similarity scores, retrieval latency, and downstream metrics like LLM answer quality).
   - Alert on significant drops in these metrics.

---

### 47. Your semantic search fails for short queries. How do you improve it?

**Why short queries fail:**
- Short queries have less semantic signal — fewer tokens means the embedding has less information to work with.
- Ambiguity: "Python" could be a programming language, a snake, or a movie.
- The query embedding may not land in the right region of the embedding space.

**Solutions:**

1. **Query expansion (LLM-based):**
   - Use an LLM to expand the query: "Python" → "Python programming language tutorials documentation".
   - Generate multiple expanded queries, embed each, and merge results.

2. **HyDE (Hypothetical Document Embeddings):**
   - Use an LLM to generate a hypothetical answer/document that would answer the query.
   - Embed the hypothetical document instead of the raw query.
   - Since documents are longer and more descriptive, the embedding is richer.

3. **Hybrid search:**
   - Short queries often contain important keywords. BM25 can catch exact matches that vector search misses.
   - Combine BM25 + vector with RRF fusion.

4. **Query reformulation:**
   - Detect short queries (< 3 tokens) and prompt the user for clarification.
   - Or automatically generate multiple interpretations and search for all of them.

5. **Contextualized queries:**
   - If there's conversation history or user context, prepend it to the query before embedding.
   - "Python" with context "user is a data scientist" → embed "Python data science".

6. **Fine-tune embedding model:**
   - If you have (short_query, relevant_document) pairs, fine-tune the embedding model specifically on short queries.

7. **Multi-vector queries:**
   - Use ColBERT-style late interaction: represent the query as multiple token-level vectors and compute fine-grained similarity against document token vectors.
   - Better at handling short, ambiguous queries because it preserves per-token information.

8. **Reranker:**
   - Retrieve more candidates (top-50 or top-100) with a permissive threshold.
   - Use a cross-encoder reranker to reorder by true relevance.
