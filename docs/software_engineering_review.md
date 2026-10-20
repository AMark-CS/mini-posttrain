# Software engineering design review

Reviewed on 2026-10-20 using the installed `using-agent-skills` workflow and the five axes from `code-review-and-quality`: correctness, readability, architecture, security, and performance. This is a source-based design review supported by the [local execution record](verification.md), not a production-readiness certification. Recommendations below are not implemented by this documentation change.

## Design assessment

The package is a useful teaching core: objectives, data preparation, LoRA, model loading, and training loops are separated into small modules. Keeping application-specific rewards and deployment outside the core is an appropriate boundary. Dataclass configurations and explicit loops make the behavior inspectable; module-level collators wrapped with `functools.partial` also accommodate Windows multiprocessing.

The weak point is the contract between these modules. Configuration values, conversation assumptions, checkpoint identity, and training progress are largely implicit. Those omissions are more consequential than the modest duplication between two short training loops. A generic training engine, plugin registry, distributed scheduler, or service layer would add complexity before solving these issues.

## Findings, ordered by impact

### High: conversation masking assumes a prefix that does not hold for multiple turns

In [SFTDataset._tokenize](../nano_posttrain/data/sft_dataset.py#L101), `prompt_messages` removes every assistant message, then uses the rendered length as a boundary in the original complete conversation. For `user1, assistant1, user2, assistant2`, the filtered prompt is not a prefix of the original conversation. This can supervise the wrong tokens or mask useful response tokens. Current tests cover a simple single-turn example, not this case.

The smallest repair is to declare and validate single-turn support if that is the intended scope. Supporting arbitrary conversations instead requires per-message assistant masks or a verified final-response prefix. Acceptance: token-level assertions for single-turn, multi-turn, system-message, truncation, and no-supervised-token cases, with the mask checked against intended roles rather than only tensor shape.

### High: adapter restoration can silently accept missing adapter tensors

[load_lora_into](../nano_posttrain/eval/eval_harness.py#L46) uses `strict=False` and rejects unexpected LoRA keys, but ignores missing adapter keys. An empty or partial state dictionary can leave adapters at initialization while logging a successful load. The harness also reconstructs default LoRA settings rather than reading rank/alpha/targets from checkpoint metadata. Custom adapter semantics are therefore not a stable serialized contract.

Validate the exact expected adapter key set and shapes and persist base-model identity/revision plus LoRA configuration. Keep this separate from full training resume, which additionally needs optimizer, scheduler, step, and RNG state. Write artifacts atomically or into a temporary directory before replacing the latest checkpoint. Acceptance: round-trip output parity and explicit failures for empty, partial, wrong-rank, and wrong-base adapters; an interrupted save must preserve the previous usable artifact.

### Medium: configuration precedence is surprising and validation is incomplete

Both [SFT CLI](../nano_posttrain/cli/train_sft.py#L43) and [DPO CLI](../nano_posttrain/cli/train_dpo.py#L45) choose the JSON dictionary instead of merging explicit command-line overrides. Unknown config keys are silently dropped, while `--data_path` remains required by argparse even with a config file. A user can supply apparently valid overrides that have no effect.

Define one precedence rule, reject unknown keys, and validate bounds before loading model weights: positive accumulation/batch/length values, valid step/epoch limits, nonempty datasets, and at least one supervised response token. Reuse one small validation/config loader only where the two CLIs actually share behavior. Acceptance: tests for config-only use, explicit override precedence, typos, empty data, and invalid length relationships.

### Medium: training progress and checkpoint names are not a resumable contract

[SFTTrainer.fit](../nano_posttrain/trainers/sft_trainer.py#L147) and [DPOTrainer.fit](../nano_posttrain/trainers/dpo_trainer.py#L195) stop at either the epoch bound or requested step bound. Partial accumulation at the end is not explicitly flushed. The first update uses zero learning rate. SFT saves under `step_latest`, DPO under `latest`; neither persists optimizer state or automatically resumes. The README now documents these limits, but wrappers cannot infer completed optimizer work from a requested step count alone.

Choose a documented epoch/step/partial-batch policy and report completed optimizer steps. Return a small structured result containing the actual checkpoint path and progress rather than requiring callers to reconstruct directory names. Do not introduce automatic resume until there is a demonstrated requirement; if added, test interruption/restart equivalence.

### Medium: package migration is incomplete and no automated quality gate is checked in

The default suite records 13 passes, one stale-path failure, and three skipped smoke tests. [Shell launchers](../scripts/run_sft.sh) still invoke `cli.*` and top-level sample paths. No GitHub Actions workflow is tracked. The source tree also tracks 25 generated `.pyc` files; ignore rules do not remove already tracked files.

Repair shipped paths together, remove tracked bytecode in a separate housekeeping change, and add one CPU job for unit tests plus CLI/import checks against an installed package. Package installation should be tested outside the checkout so `sys.path` injection does not hide packaging errors. Keep model-dependent tests opt-in or separately scheduled. Add one tiny-model trainer test that verifies parameters change and saved adapters restore, rather than many tests mirroring implementation details.

### Medium: the TRL helper does not isolate the measured workload

[compare_with_trl.py](../scripts/compare_with_trl.py#L41) constructs a trainer, which builds its model in `__init__`, then calls `_build()` again. CUDA memory-stat reset is unconditional, timing includes loading, and the output omits final loss. There is no controlled result supporting comparative speed claims.

Remove the second build, separate warm-up/loading from steady-state timing, synchronize CUDA around measurements, and match tokenized examples, update counts, trainable parameters, and dtype. Record peak allocated/reserved memory and objective values. Prefer a small explicit comparison script over integrating a benchmark framework.

### Medium: all-linear target selection includes a root `lm_head`

[_resolve_targets](../nano_posttrain/optim/lora.py#L64) excludes names ending in `.lm_head`, which does not exclude the root module name `lm_head`. That contradicts the nearby comment and can increase adapter size and affect output-head behavior. Decide the intended target policy, test both root and nested names, and retain target metadata with adapters. The reported 169 modules describe the actual run and should not be retroactively changed.

## Five-axis summary

| Axis | What works | Main gap and next check |
| --- | --- | --- |
| Correctness | Loss unit tests and short SFT/DPO execution exist | Multi-turn masks, adapter completeness, partial accumulation, and configuration contracts need behavioral tests |
| Readability | Small explicit modules and loops | Repair stale launch paths/comments before creating new abstractions |
| Architecture | Reusable core is separate from e-commerce application | Make checkpoint, configuration, and trainer-result contracts explicit |
| Security | No hosted request surface in this core | Treat checkpoint files as trusted inputs; inspect and constrain the actual deserialization behavior across supported torch versions and validate tensor contents |
| Performance | LoRA and CPU-reference mode are available | Benchmark actual workloads; full-vocabulary logits, eager tokenization, and CPU reference passes can dominate memory/time |

This review did not perform dependency vulnerability scanning or a hostile-checkpoint security test. No performance improvement is inferred from a two-step run.

## Smallest useful improvement sequence

1. Restore a green unit baseline and working installed-package CLI paths. Add masking and adapter-completeness regression cases.
2. Correct masking/adapter validation and document one configuration-precedence rule.
3. Standardize checkpoint metadata and return actual training progress; preserve separate SFT/DPO loops until a concrete shared change justifies extraction.
4. Add a CPU CI job and isolated packaging check, then collect one controlled benchmark before optimizing.

The recommended quality bar is observable: no invalid supervised masks, no silent partial adapter load, deterministic config interpretation, a passing CPU suite, and reproducible artifact identities. Distributed training, web services, and generalized plugin systems are outside this core's demonstrated need.
