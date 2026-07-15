# OLMo-3 Judge Backend (Tasks 1–2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local, open, no-API-key **OLMo-3-7B chat judge** as a third backend (`--backend olmo`) to `src/judge_results.py`, reusing the existing `run_steer`/`run_interpret` machinery so OLMo gets both modes for free.

**Architecture:** OLMo-3-7B-Instruct is a *general chat judge*, not a narrow yes/no head, so it can consume the existing `STEER_SYS`/`INTERPRET_SYS` JSON prompts. Task 1 builds a thin `OlmoJudge` wrapper whose `.chat(system, user, max_tokens)` mirrors the Anthropic `ask()` contract. Task 2 makes `judge_results.ask()` dispatch to any client exposing `.chat` (duck-typed), and wires `--backend olmo` through the *unchanged* `run_steer`/`run_interpret`. The Anthropic and `local-hf` paths are untouched.

**Tech Stack:** Python 3.13 (`.venv`), transformers 5.12.1, torch 2.12.0 (MPS), pytest. Model: `allenai/Olmo-3-7B-Instruct` (fp16 ≈14 GB, fits the M5 Pro's 24 GB unified memory).

## Global Constraints

- **Device:** default judge device is `cuda` in argparse for cluster compatibility, but on this machine pass `--device mps`. Do not hardcode `mps` in the module — it is a constructor/CLI argument.
- **transformers 5.12.1:** use the modern `dtype=torch.float16` kwarg on `from_pretrained` (not the deprecated `torch_dtype=`). Verified both exist in this version; `dtype` is forward-compatible.
- **No model download in unit tests.** Every test in this plan must pass without network access and without instantiating `OlmoJudge` (which would download 7B weights). The real model is exercised only by an env-gated smoke test (`RUN_OLMO_SMOKE=1`) and by the manual Task-3 runs (out of scope here).
- **Downloads:** prefix any real model run with `HF_HUB_DISABLE_XET=1` (per project memory).
- **Package layout:** `src/judges/` is an importable package (`__init__.py` exists). Tests add `src/` to `sys.path` via `sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))` — follow that exact pattern (matches `tests/test_local_hf.py`).
- **Backend discriminator:** `ask()` routes to a local chat judge via `hasattr(client, "chat")`. The Anthropic client exposes `.messages`, not `.chat`, so this is an unambiguous discriminator and needs no `isinstance` import of the heavy module.
- **Run tests with:** `.venv/bin/python -m pytest <path> -v` from the repo root.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `src/judges/olmo_judge.py` | **Create.** `OlmoJudge` wrapper: loads OLMo-3 on a device, `_build_messages` (pure, testable), `.chat()` (model call). | 1 |
| `tests/test_olmo_judge.py` | **Create.** Unit-tests `_build_messages`; env-gated smoke test for the real model. | 1 |
| `src/judge_results.py` | **Modify.** `ask()` → duck-typed `.chat` dispatch; extract `build_parser()`; add `olmo` to `--backend`, add `--olmo-model`; add `olmo` branch in `main()`. | 2 |
| `tests/test_judge_backend.py` | **Create.** Unit-tests `ask()` dispatch to a fake chat client and `build_parser()` accepting `--backend olmo`. | 2 |

---

## Task 1: `OlmoJudge` chat-judge module

**Files:**
- Create: `src/judges/olmo_judge.py`
- Test: `tests/test_olmo_judge.py`

**Interfaces:**
- Consumes: nothing from earlier tasks. Imports `torch`, `transformers.AutoModelForCausalLM`, `transformers.AutoTokenizer`.
- Produces:
  - `DEFAULT_OLMO = "allenai/Olmo-3-7B-Instruct"` (module constant).
  - `class OlmoJudge`:
    - `__init__(self, model=DEFAULT_OLMO, device="mps")` — loads tokenizer + model, sets `self.device`, `self.model_name`, `self.tok`, `self.model`.
    - `@staticmethod _build_messages(system: str, user: str) -> list[dict]` — returns `[{"role":"system","content":system},{"role":"user","content":user}]`. **Pure, no model needed.**
    - `chat(self, system: str, user: str, max_tokens: int = 200) -> str` — returns the model's generated text (prompt stripped), suitable for `extract_json`.
  - Task 2's `main()` constructs `OlmoJudge(args.olmo_model, args.device)` and passes it where `run_steer`/`run_interpret` expect a `client`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_olmo_judge.py`:

```python
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from judges.olmo_judge import OlmoJudge, DEFAULT_OLMO


# _build_messages is pure (no model load), so it is the unit-testable surface of the judge.
# Constructing a real OlmoJudge downloads a 7B model, so that path is only exercised by the
# env-gated smoke test below (run manually on the M5 Pro).
def test_build_messages_has_system_then_user():
    msgs = OlmoJudge._build_messages("SYS", "USER")
    assert msgs == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]


def test_default_model_is_olmo_instruct():
    assert DEFAULT_OLMO == "allenai/Olmo-3-7B-Instruct"


@pytest.mark.skipif(not os.environ.get("RUN_OLMO_SMOKE"),
                    reason="downloads and runs the 7B OLMo model; set RUN_OLMO_SMOKE=1 to enable")
def test_olmo_smoke_true_false():
    from judge_results import STEER_SYS, extract_json
    j = OlmoJudge(device=os.environ.get("OLMO_DEVICE", "mps"))
    tokyo = extract_json(j.chat(STEER_SYS, 'Stem: "The capital of Japan is"\nCompletion: "Tokyo"', 80))
    three = extract_json(j.chat(STEER_SYS, 'Stem: "Two plus two equals"\nCompletion: "three"', 80))
    assert tokyo and tokyo.get("verdict") == "TRUE"
    assert three and three.get("verdict") == "FALSE"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_olmo_judge.py -v`
Expected: FAIL at collection/import with `ModuleNotFoundError: No module named 'judges.olmo_judge'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/judges/olmo_judge.py`:

```python
"""Local OLMo-3 chat judge (open weights, no API key).

Mirrors the Anthropic `ask()` contract in judge_results.py: `.chat(system, user, max_tokens)`
returns the model's text, from which `extract_json` pulls the verdict. Because OLMo-3-Instruct is
a general instruction-following chat model (not a narrow yes/no head), it drives BOTH the `steer`
and `interpret` prompts unchanged.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_OLMO = "allenai/Olmo-3-7B-Instruct"


class OlmoJudge:
    def __init__(self, model=DEFAULT_OLMO, device="mps"):
        self.device = device
        self.model_name = model
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, dtype=torch.float16).to(device).eval()

    @staticmethod
    def _build_messages(system, user):
        return [{"role": "system", "content": system},
                {"role": "user", "content": user}]

    def chat(self, system, user, max_tokens=200):
        # transformers 5.x returns a BatchEncoding (input_ids + attention_mask) here, so pass the
        # dict through to generate() and slice the prompt off using input_ids' length.
        inputs = self.tok.apply_chat_template(
            self._build_messages(system, user),
            add_generation_prompt=True, return_tensors="pt", return_dict=True).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        prompt_len = inputs["input_ids"].shape[1]
        return self.tok.decode(out[0][prompt_len:], skip_special_tokens=True)
```

> **Note (verified 2026-07-14):** an earlier draft used `return_tensors="pt"` without
> `return_dict=True` and sliced `ids.shape` — that raises `AttributeError` on transformers 5.12.1,
> which returns a `BatchEncoding` (not a bare tensor) from `apply_chat_template`. The code above is
> the fix and was validated end-to-end (chat → clean JSON → `extract_json`) using a small stand-in
> chat model, since the real 7B model swap-thrashes this 24 GB machine when other apps are resident.
> The OLMo-3-7B run itself belongs on the cluster or a freed-up machine. The OLMo-3 chat template
> **does** accept a `system` role, so the system-fold fallback below was not needed.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_olmo_judge.py -v`
Expected: PASS for `test_build_messages_has_system_then_user` and `test_default_model_is_olmo_instruct`; `test_olmo_smoke_true_false` SKIPPED (`RUN_OLMO_SMOKE` unset).

- [ ] **Step 5: Commit**

```bash
git add src/judges/olmo_judge.py tests/test_olmo_judge.py
git commit -m "feat: OlmoJudge local chat-judge wrapper (task 1)"
```

**Deferred verification (manual, on the M5 Pro — not part of the automated suite):**

```bash
RUN_OLMO_SMOKE=1 OLMO_DEVICE=mps HF_HUB_DISABLE_XET=1 \
  .venv/bin/python -m pytest tests/test_olmo_judge.py::test_olmo_smoke_true_false -v -s
```
Expected: PASS (`Tokyo → TRUE`, `three → FALSE`). **Risk to check here:** if `apply_chat_template` raises because the OLMo-3 template rejects a `system` role, fold the system text into the user turn — change `_build_messages` to return a single user message `f"{system}\n\n{user}"` and re-run. Confirm the JSON still parses.

---

## Task 2: Wire `--backend olmo` into `judge_results.py`

**Files:**
- Modify: `src/judge_results.py` (`ask()` at lines 45–56; `main()` at lines 241–264)
- Test: `tests/test_judge_backend.py`

**Interfaces:**
- Consumes: `OlmoJudge` from Task 1 (constructed lazily inside `main()`; `ask()` does NOT import it — it duck-types on `.chat`).
- Produces:
  - `ask(client, model, system, user, max_tokens=200, retries=4)` — if `hasattr(client, "chat")`, returns `client.chat(system, user, max_tokens)`; otherwise runs the existing Anthropic retry loop. Signature unchanged.
  - `build_parser() -> argparse.ArgumentParser` — extracted from `main()`; `--backend` choices are `["anthropic", "local-hf", "olmo"]`; adds `--olmo-model` (default `allenai/Olmo-3-7B-Instruct`).
  - `main()` gains an `elif args.backend == "olmo":` branch that constructs `OlmoJudge` and calls `run_steer`/`run_interpret`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_judge_backend.py`:

```python
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from judge_results import ask, build_parser


class _FakeChat:
    """Stands in for OlmoJudge: exposes .chat, records the call. No model download."""
    def __init__(self):
        self.calls = []

    def chat(self, system, user, max_tokens=200):
        self.calls.append((system, user, max_tokens))
        return '{"verdict": "TRUE", "reason": "ok"}'


def test_ask_routes_chat_client_to_chat():
    fake = _FakeChat()
    out = ask(fake, "ignored-model", "SYS", "USER", max_tokens=80)
    assert out == '{"verdict": "TRUE", "reason": "ok"}'
    assert fake.calls == [("SYS", "USER", 80)]


def test_parser_accepts_olmo_backend_and_default_model():
    args = build_parser().parse_args(
        ["--mode", "steer", "--dataset", "cities", "--backend", "olmo"])
    assert args.backend == "olmo"
    assert args.olmo_model == "allenai/Olmo-3-7B-Instruct"


def test_parser_rejects_unknown_backend():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["--mode", "steer", "--dataset", "cities", "--backend", "nope"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_judge_backend.py -v`
Expected: FAIL at import — `ImportError: cannot import name 'build_parser' from 'judge_results'`.

- [ ] **Step 3a: Make `ask()` dispatch on `.chat`**

In `src/judge_results.py`, replace the `ask` function (lines 45–56) with:

```python
def ask(client, model, system, user, max_tokens=200, retries=4):
    """One judge call; returns the text response.

    Local chat judges (e.g. OlmoJudge) expose a `.chat(system, user, max_tokens)` method and are
    dispatched directly. The Anthropic client has no `.chat`, so it falls through to the retry loop.
    """
    if hasattr(client, "chat"):
        return client.chat(system, user, max_tokens)
    for attempt in range(retries):
        try:
            msg = client.messages.create(
                model=model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": user}])
            return msg.content[0].text
        except Exception as e:  # noqa: BLE001 - transient API errors → retry
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
```

- [ ] **Step 3b: Extract `build_parser()` and add the `olmo` branch to `main()`**

In `src/judge_results.py`, replace `main()` (lines 241–264) with:

```python
def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["steer", "interpret"])
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--limit", type=int, default=0, help="only first N items (smoke test)")
    p.add_argument("--backend", choices=["anthropic", "local-hf", "olmo"], default="anthropic",
                   help="anthropic = Claude API; local-hf = TruthfulQA HF judges; "
                        "olmo = local OLMo-3 chat judge (open, no API key)")
    p.add_argument("--device", default="cuda", help="device for local judges (use 'mps' on Mac)")
    p.add_argument("--olmo-model", default="allenai/Olmo-3-7B-Instruct",
                   help="model id for --backend olmo")
    return p


def main():
    args = build_parser().parse_args()
    if args.backend == "anthropic":
        client = get_client()
        if args.mode == "steer":
            run_steer(client, args.model, args.dataset, args.limit)
        else:
            run_interpret(client, args.model, args.dataset, args.limit)
    elif args.backend == "olmo":
        sys.path.insert(0, os.path.dirname(__file__))
        from judges.olmo_judge import OlmoJudge
        client = OlmoJudge(args.olmo_model, args.device)
        if args.mode == "steer":
            run_steer(client, args.olmo_model, args.dataset, args.limit)
        else:
            run_interpret(client, args.olmo_model, args.dataset, args.limit)
    else:  # local-hf
        sys.path.insert(0, os.path.dirname(__file__))
        from judges.local_hf import get_judge
        judge = get_judge(args.dataset, args.device)
        if args.mode == "steer":
            run_steer_local(judge, args.dataset, args.limit)
        else:
            raise SystemExit("local-hf interpret mode not supported; use --backend anthropic or olmo")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_judge_backend.py -v`
Expected: PASS (all three tests).

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: all previously-green tests still pass, plus the new `test_olmo_judge.py` and `test_judge_backend.py` tests (smoke test SKIPPED).

- [ ] **Step 6: Verify `--help` shows the new backend**

Run: `.venv/bin/python src/judge_results.py --help`
Expected: `--backend {anthropic,local-hf,olmo}` and `--olmo-model` appear; no import of torch/OLMo happens (lazy).

- [ ] **Step 7: Commit**

```bash
git add src/judge_results.py tests/test_judge_backend.py
git commit -m "feat: wire --backend olmo through run_steer/run_interpret (task 2)"
```

---

## Self-Review

**Spec coverage (OLMO plan Tasks 1–2):**
- Task 1 "OLMo chat-judge module" → this plan's Task 1 (`olmo_judge.py`, `OlmoJudge.chat`). ✓
- Task 2 "Wire `--backend olmo`" → Task 2 Steps 1–3 (argparse `olmo` + `--olmo-model`, `ask` dispatch, `main` branch, both modes reused). ✓ (The OLMO doc's Step 2 used `isinstance(client, OlmoJudge)`; this plan uses `hasattr(client, "chat")` — same behavior, but unit-testable without downloading the model and without importing the heavy module inside `ask`.)

**Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to" — every code step shows complete code. ✓

**Type consistency:** `OlmoJudge(model, device)`, `.chat(system, user, max_tokens)`, `_build_messages(system, user)`, `DEFAULT_OLMO`, `build_parser()`, `ask(client, model, system, user, max_tokens, retries)` — names identical across Task 1, Task 2, and both test files. `main()` passes `args.olmo_model` as the `model` arg to `run_steer`/`run_interpret`, which only use it for print labels and forward it to `ask()`, where `.chat` ignores it — consistent. ✓

**Out of scope (deferred to the OLMO plan doc):** Task 3 (running the quick-win on cities/common_claim), Task 4 (`validate_judge.py`), Tasks 5–6 (Think variant, Dolci safety subset). These need the downloaded model and are run manually on the M5 Pro.
