# Research Guide

The application includes an experimental verification layer for studying constrained generation and repair in multi-turn psychological training dialogues.

## RAVR Pipeline

RAVR follows a verification-and-repair loop:

1. evaluate the student's therapist turn;
2. retrieve methodology-specific rules;
3. produce a structured proof object;
4. identify violated constraints;
5. generate a targeted repair when enabled;
6. re-verify the repaired turn;
7. retain turn-level metrics and evidence.

The proof object includes adherence, satisfied and violated constraints, evidence, recommendations, retrieved chunks, citations, citation diagnostics, and repair results.

## RAVR-S Extension

RAVR-S adds state-sensitive candidate selection. It uses predicted changes in interaction state when ranking candidate responses, while retaining methodology verification and targeted repair. The extension is disabled by default because experimental comparisons should opt into it explicitly.

```env
RAVRS_ENABLE=1
RAVRS_K=3
```

See [Configuration](CONFIGURATION.md) for all controls.

## Reproducible Benchmark Request

```bash
curl -X POST http://127.0.0.1:8000/api/ravr_benchmark \
  -H 'Content-Type: application/json' \
  -d '{
    "n_per_case": 6,
    "random_seed": 42,
    "include_llm_eval": false,
    "disable_eval_cache": false
  }' > benchmark.json
```

For provider comparisons:

```bash
curl -X POST http://127.0.0.1:8000/api/ravr_multi_model_benchmark \
  -H 'Content-Type: application/json' \
  -d '{
    "providers": ["openrouter_gpt", "openrouter_deepseek", "openrouter_qwen"],
    "n_per_case": 4,
    "random_seed": 42,
    "include_llm_eval": true,
    "llm_temperature": 0.2
  }' > multi_model.json
```

## Offline Research Artifacts

Large experiment runners, raw generations, paper sources, and generated reports are intentionally excluded from the application repository. Publish experiment-specific code and outputs as a versioned reproducibility package rather than mixing them with the deployable simulator.

The HTTP benchmark endpoints remain part of the application and are sufficient for exporting matched JSON, CSV, and JSONL records for an external analysis pipeline.

## Minimum Run Metadata

Every reported run should record:

- repository commit hash;
- execution date and timezone;
- dataset or case identifiers;
- random seed;
- provider and returned model identifier;
- temperature, top-p, and sample count;
- exact prompts or prompt hashes;
- all RAVR and RAVR-S switches;
- cache behavior;
- parser failures and excluded records;
- API call counts and estimated cost;
- raw model output or a documented compact representation;
- analysis-script version.

## Evaluation Boundaries

Verifier scores are operational research measurements, not clinical judgments. They depend on the encoded methodology rules, evaluator prompts, provider behavior, and parser. Report human evaluation and domain review separately, and do not describe automated labels as expert ground truth.

Likewise, the bundled cases are synthetic training artifacts. They are not prevalence estimates, diagnostic instruments, or substitutes for supervised clinical education.
