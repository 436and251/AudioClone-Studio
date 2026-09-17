# AudioClone Studio Training Workspace and Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the training-module boundary to protocol v2, replace stage-shaped tabs with a stable three-page training workspace, and run one-click inference only with a human-promoted model.

**Architecture:** AudioMiner remains a lightweight host that renders descriptor-driven controls and launches isolated module subprocesses. voice-pipeline owns GPT-SoVITS data parsing, model promotion, and inference; the repositories communicate through strict protocol-v2 JSON files and JSONL events. Training, promotion, and inference remain mutually exclusive subprocess operations.

**Tech Stack:** Python 3.12, PySide6 QtWidgets/QtMultimedia, JSON/JSONL, Typer, pytest, existing voice-pipeline inference runtime.

**Spec:** `docs/superpowers/specs/2026-09-17-training-workspace-inference-design.md`

## Global Constraints

- Repositories: `D:/Python_program_codes/AudioMiner(voice-clone)` and `D:/AI-Training/voice-clone/voice-pipeline/voice-pipeline`.
- Work directly on each `main` branch; do not create a worktree or another branch.
- Before each task, send a short Chinese `Tips` summary; stop after each task for user review.
- Never stage or modify AudioMiner `.idea/` or unrelated user changes.
- AudioMiner must not import voice-pipeline, Torch, or model code.
- Training modules use their configured absolute project root and Python executable in isolated subprocesses.
- Protocol v2 rejects protocol v1 explicitly; no silent field guessing.
- Training, promotion, and inference never run concurrently in one workspace.
- All ordinary visible text is localized in `zh_CN`, `en`, and `ja`.
- Do not add GUI or runtime dependencies.
- Do not generate EXE, ZIP, model copies, or additional training weights.
- Use `D:/Python_program_codes/TTS-Inference/.venv-gpt-sovits/Scripts/python.exe` for complete development dependencies.

---

### Task 1: voice-pipeline protocol-v2 descriptor and generic training data

**Repository:** `D:/AI-Training/voice-clone/voice-pipeline/voice-pipeline`

**Files:**
- Modify: `src/voice_pipeline/module_api/descriptor.py`
- Modify: `src/voice_pipeline/module_api/job.py`
- Modify: `src/voice_pipeline/module_api/materialize.py`
- Modify: `tests/test_module_descriptor.py`
- Modify: `tests/test_module_job.py`
- Modify: `tests/test_module_materialize.py`

**Interfaces:**
- Produces descriptor root `protocol_version == 2`.
- Produces framework field `training_data: {kind: "file", extensions: [".list"]}` and capability `infer`.
- Produces `ModuleTrainingData(path: Path, kind: str)` and `ModuleJob.training_data`.
- Consumes job fields `module_id`, `framework`, and `training_data` instead of `dataset_list`/`dataset_sha256`.

- [ ] **Step 1: Write failing descriptor tests**

```python
payload = build_descriptor()
framework = payload["frameworks"][0]
assert payload["protocol_version"] == 2
assert framework["training_data"] == {"kind": "file", "extensions": [".list"]}
assert framework["capabilities"] == [
    "preprocess", "train", "evaluate", "listen", "promote", "infer"
]
```

- [ ] **Step 2: Write failing ModuleJob tests**

Use the v2 fixture fields below, then cover file acceptance, directory/extension rejection, containment, module mismatch, unknown nested fields, and explicit v1 rejection.

```python
"protocol_version": 2,
"module_id": "gpt-sovits-v2proplus",
"training_data": {"path": str(dataset.resolve()), "kind": "file"},
```

- [ ] **Step 3: Run RED tests**

```powershell
python -m pytest tests/test_module_descriptor.py tests/test_module_job.py tests/test_module_materialize.py -q
```

Expected: protocol-version, descriptor-field, and removed dataset attribute failures.

- [ ] **Step 4: Implement the v2 adapter**

```python
@dataclass(frozen=True, slots=True)
class ModuleTrainingData:
    path: Path
    kind: str
```

Validate the generic shape against the selected framework descriptor. For GPT-SoVITS, `materialize_job()` passes `job.training_data.path` into the existing manifest configuration and `_training_languages()` reads the same path. Format parsing stays inside voice-pipeline.

- [ ] **Step 5: Verify**

```powershell
python -m pytest tests/test_module_descriptor.py tests/test_module_job.py tests/test_module_materialize.py tests/test_module_runner.py tests/test_orchestrator.py -q
```

- [ ] **Step 6: Commit**

```powershell
git add src/voice_pipeline/module_api/descriptor.py src/voice_pipeline/module_api/job.py src/voice_pipeline/module_api/materialize.py tests/test_module_descriptor.py tests/test_module_job.py tests/test_module_materialize.py
git commit -m "feat: upgrade training module contract to v2"
```

---

### Task 2: AudioMiner protocol-v2 parsing and generic training-data selector

**Repository:** `D:/Python_program_codes/AudioMiner(voice-clone)`

**Files:**
- Modify: `tts_builder/training_modules/models.py`
- Modify: `tts_builder/training_modules/protocol.py`
- Modify: `tts_builder/training_modules/job.py`
- Modify: `tts_builder/gui/training_form.py`
- Modify: `tts_builder/gui/i18n.py`
- Modify: `tests/training_modules/test_protocol.py`
- Modify: `tests/training_modules/test_job.py`
- Modify: `tests/gui/test_training_form.py`

**Interfaces:**
- Produces `TrainingDataDescriptor(kind: str, extensions: tuple[str, ...])` on `FrameworkDescriptor`.
- Produces `TrainingForm.training_data_path() -> Path`.
- Produces protocol-v2 `job.json` with `module_id` and `training_data`.
- Keeps `prefill_dataset(path)` as the mining bridge while displaying “Training data”.

- [ ] **Step 1: Write failing parser tests**

Accept file/directory descriptors. Reject unknown keys, unsupported kinds, extensions on directories, malformed suffixes, and protocol version 1 with `unsupported module protocol version 1; expected 2`.

- [ ] **Step 2: Write failing job tests**

```python
assert payload["protocol_version"] == 2
assert payload["module_id"] == descriptor.module_id
assert payload["training_data"] == {
    "path": str(training_path.resolve()),
    "kind": "file",
}
assert "dataset_list" not in payload
assert "dataset_sha256" not in payload
```

Also cover directories, wrong type/extension, missing input, traversal, and symlink escape.

- [ ] **Step 3: Write failing form tests**

Assert `训练数据` / `Training data` / `トレーニングデータ`. A file descriptor calls `getOpenFileName`; a directory descriptor calls `getExistingDirectory`. Validation changes with the selected framework.

- [ ] **Step 4: Run RED tests**

```powershell
python -m pytest tests/training_modules/test_protocol.py tests/training_modules/test_job.py tests/gui/test_training_form.py -q
```

- [ ] **Step 5: Implement the host parser, job, and one generic selector**

Change only browse behavior and validation from the descriptor. Do not create framework-specific widgets.

- [ ] **Step 6: Verify focused tests and the real handshake**

```powershell
$env:VOICE_PIPELINE_PYTHON='D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe'
$env:VOICE_PIPELINE_ROOT='D:\AI-Training\voice-clone\voice-pipeline\voice-pipeline'
python -m pytest tests/training_modules tests/gui/test_training_form.py tests/integration/test_voice_pipeline_module.py -q
```

- [ ] **Step 7: Commit**

```powershell
git add tts_builder/training_modules/models.py tts_builder/training_modules/protocol.py tts_builder/training_modules/job.py tts_builder/gui/training_form.py tts_builder/gui/i18n.py tests/training_modules/test_protocol.py tests/training_modules/test_job.py tests/gui/test_training_form.py
git commit -m "feat: accept generic training data from modules"
```

---

### Task 3: Stable training workspace and visual hierarchy

**Repository:** `D:/Python_program_codes/AudioMiner(voice-clone)`

**Files:**
- Modify: `tts_builder/gui/navigation.py`
- Modify: `tts_builder/gui/styles.py`
- Modify: `tts_builder/gui/training_page.py`
- Modify: `tts_builder/gui/candidate_page.py`
- Modify: `tts_builder/gui/i18n.py`
- Modify: `tests/gui/test_adaptive_shell.py`
- Modify: `tests/gui/test_training_page.py`
- Modify: `tests/gui/test_candidate_page.py`

**Interfaces:**
- Fixed tab IDs are `config`, `candidates`, `inference`.
- `CandidatePage.set_candidates(())` renders a localized empty state.
- The inference tab is a placeholder only until Task 6 replaces it with `InferencePage`.

- [ ] **Step 1: Write failing navigation tests**

After applying `APP_QSS`, assert `NavigationTitle` has a larger font and heavier weight than `NavigationButton`; compact mode hides the title.

- [ ] **Step 2: Write failing fixed-tab and localization tests**

```python
assert [page.tabs.tabText(i) for i in range(3)] == [
    "Training configuration", "Candidate listening", "Inference trial"
]
```

Repeat after Chinese and Japanese changes. Assert form/progress are only in page 0 and candidates only in page 1.

- [ ] **Step 3: Write the candidate empty-state test**

Without a listening manifest, assert the localized empty hint is visible and cards/promotion are disabled.

- [ ] **Step 4: Run RED tests**

```powershell
python -m pytest tests/gui/test_adaptive_shell.py tests/gui/test_training_page.py tests/gui/test_candidate_page.py -q
```

- [ ] **Step 5: Implement the three-page layout**

Move current form/actions/progress/errors/Activity into `config_page`, add the existing candidate page as page 1, and an empty QWidget as page 2. Capabilities change page state, not page existence.

- [ ] **Step 6: Verify and commit**

```powershell
python -m pytest tests/gui/test_adaptive_shell.py tests/gui/test_training_page.py tests/gui/test_candidate_page.py -q
git add tts_builder/gui/navigation.py tts_builder/gui/styles.py tts_builder/gui/training_page.py tts_builder/gui/candidate_page.py tts_builder/gui/i18n.py tests/gui/test_adaptive_shell.py tests/gui/test_training_page.py tests/gui/test_candidate_page.py
git commit -m "refactor: organize the training workspace by user task"
```

---

### Task 4: Collapsible advanced settings, wheel safety, and immutable run snapshots

**Repository:** `D:/Python_program_codes/AudioMiner(voice-clone)`

**Files:**
- Modify: `tts_builder/gui/training_form.py`
- Modify: `tts_builder/gui/training_page.py`
- Modify: `tts_builder/gui/styles.py`
- Modify: `tts_builder/gui/i18n.py`
- Modify: `tests/gui/test_training_form.py`
- Modify: `tests/gui/test_training_page.py`

**Interfaces:**
- Produces frozen `TrainingSelection` from `TrainingForm.snapshot()`.
- The active process consumes only the Start-time snapshot.
- Form fields remain enabled while running; Start is disabled and `next_run_hint` is visible.

- [ ] **Step 1: Write failing collapse tests**

Assert advanced content starts hidden, the title toggles it, no bottom border is applied, and framework changes do not force it open.

- [ ] **Step 2: Write failing wheel tests**

Send wheel events to common and advanced combo/spin widgets. Assert values remain unchanged while the containing scroll area can scroll.

- [ ] **Step 3: Write failing snapshot tests**

Start a fake process, edit batch size and project name, and assert the built job kept original values while the visible form keeps new values. Start remains disabled, Stop enabled, and the localized next-run hint visible.

- [ ] **Step 4: Run RED tests**

```powershell
python -m pytest tests/gui/test_training_form.py tests/gui/test_training_page.py -q
```

- [ ] **Step 5: Implement with native Qt only**

Use a checkable `QToolButton` and content QWidget. Install one local event filter on combo/spin widgets. Use one frozen dataclass for the snapshot.

- [ ] **Step 6: Verify and commit**

```powershell
python -m pytest tests/gui/test_training_form.py tests/gui/test_training_page.py -q
git add tts_builder/gui/training_form.py tts_builder/gui/training_page.py tts_builder/gui/styles.py tts_builder/gui/i18n.py tests/gui/test_training_form.py tests/gui/test_training_page.py
git commit -m "feat: make training configuration safe to edit"
```

---

### Task 5: voice-pipeline protocol inference command

**Repository:** `D:/AI-Training/voice-clone/voice-pipeline/voice-pipeline`

**Files:**
- Create: `src/voice_pipeline/module_api/infer.py`
- Modify: `src/voice_pipeline/cli/module.py`
- Create: `tests/test_module_infer.py`
- Modify: `tests/test_module_cli.py`

**Interfaces:**
- Produces `ModuleInferenceRequest.load(path: Path) -> ModuleInferenceRequest`.
- Produces `run_module_inference(request_path: Path, stream: TextIO, diagnostics: TextIO) -> Path`.
- Adds `voice-pipeline module infer --request PATH --events-jsonl`.
- Emits `inference_started`, `artifact(type="inference_audio")`, `inference_completed`, or `inference_failed`.

- [ ] **Step 1: Write strict request tests**

Create a request below `jobs/<job_id>/inference/<request_id>/request.json`. Cover protocol 2, exact fields, safe IDs, contained paths, UTF-8 TXT, exactly one text source, languages `{zh, ja, en, mixed}`, `.wav` output, and model equality with the source job's latest valid `promoted_model` artifact.

- [ ] **Step 2: Write runner tests at existing inference seams**

Monkeypatch `InferenceSession.load` and `run_synthesis_job`. Assert forwarding of model/text/language/device/output, JSONL-only stdout, contained `inference_audio`, and failure events that preserve request and input files.

- [ ] **Step 3: Write CLI tests**

Assert `--events-jsonl` is mandatory, diagnostics use stderr, and descriptor probing still does not import Torch.

- [ ] **Step 4: Run RED tests**

```powershell
python -m pytest tests/test_module_infer.py tests/test_module_cli.py tests/test_infer_cli.py -q
```

- [ ] **Step 5: Implement by reusing the inference runtime**

`run_module_inference()` loads the v2 `ModuleJob`, validates the journal's promoted model, calls `InferenceSession.load()` and `run_synthesis_job()`, and emits the result. Do not duplicate text chunking, synthesis validation, or WAV writing.

- [ ] **Step 6: Verify and commit**

```powershell
python -m pytest tests/test_module_infer.py tests/test_module_cli.py tests/test_infer_cli.py tests/test_inference_job.py tests/test_inference_session.py -q
git add src/voice_pipeline/module_api/infer.py src/voice_pipeline/cli/module.py tests/test_module_infer.py tests/test_module_cli.py
git commit -m "feat: expose promoted-model inference through module protocol"
```

---

### Task 6: AudioMiner inference request, page, playback, and result toast

**Repository:** `D:/Python_program_codes/AudioMiner(voice-clone)`

**Files:**
- Create: `tts_builder/training_modules/inference.py`
- Modify: `tts_builder/training_modules/process.py`
- Create: `tts_builder/gui/inference_page.py`
- Modify: `tts_builder/gui/training_page.py`
- Modify: `tts_builder/gui/i18n.py`
- Create: `tests/training_modules/test_inference.py`
- Create: `tests/gui/test_inference_page.py`
- Modify: `tests/training_modules/test_process.py`
- Modify: `tests/gui/test_training_page.py`

**Interfaces:**
- Produces `build_inference_request(job_path, promoted_model, text, text_file, language, device, now) -> Path`.
- Adds `ModuleProcessController.infer(request_path: Path) -> None`.
- Produces `InferencePage.set_model(path: Path | None)`, `set_busy(bool)`, and `inference_requested(InferenceInput)`.
- Consumes `inference_audio` with the existing `AudioPlayer`.

- [ ] **Step 1: Write request-builder security tests**

Assert external UTF-8 TXT is snapshotted to the request directory, inline/TXT exclusivity, model containment, output path `outputs/<project>/gui/<timestamp>.wav`, numeric suffix on collision, and atomic writes.

- [ ] **Step 2: Write process command tests**

```python
assert command == [
    python, "-m", module_name, "module", "infer",
    "--request", str(request), "--events-jsonl",
]
```

Assert the same isolated environment, bounded log polling, cancellation, and single-operation guard as run/promote.

- [ ] **Step 3: Write inference-page tests**

Assert source exclusivity, editable input without a model, disabled Start and localized hint, four languages, successful playback, Open Directory, and a non-modal path toast that auto-hides.

- [ ] **Step 4: Write coordinator tests**

Feed `promoted_model` to unlock execution and `inference_audio` to load the result. While any operation runs, candidate/infer execution is disabled while training form fields stay editable.

- [ ] **Step 5: Run RED tests**

```powershell
python -m pytest tests/training_modules/test_inference.py tests/training_modules/test_process.py tests/gui/test_inference_page.py tests/gui/test_training_page.py -q
```

- [ ] **Step 6: Implement with existing Qt components**

Use `QPlainTextEdit`, one TXT path row, a language combo, existing `AudioPlayer`, and `QLabel`/`QTimer` for the toast. Do not add dependencies.

- [ ] **Step 7: Verify and commit**

```powershell
python -m pytest tests/training_modules/test_inference.py tests/training_modules/test_process.py tests/gui/test_inference_page.py tests/gui/test_training_page.py -q
git add tts_builder/training_modules/inference.py tts_builder/training_modules/process.py tts_builder/gui/inference_page.py tts_builder/gui/training_page.py tts_builder/gui/i18n.py tests/training_modules/test_inference.py tests/training_modules/test_process.py tests/gui/test_inference_page.py tests/gui/test_training_page.py
git commit -m "feat: add promoted-model inference trial"
```

---

### Task 7: Recover the latest promoted model from durable events

**Repository:** `D:/Python_program_codes/AudioMiner(voice-clone)`

**Files:**
- Create: `tts_builder/training_modules/recovery.py`
- Modify: `tts_builder/gui/training_page.py`
- Create: `tests/training_modules/test_recovery.py`
- Modify: `tests/gui/test_training_page.py`

**Interfaces:**
- Produces `latest_promoted_model(project_root: Path, module_id: str, framework: str, project_name: str) -> Path | None`.
- Reads only contained `jobs/*/job.json` and `events.jsonl`.
- TrainingPage invokes recovery when project root, project name, or framework changes while idle.

- [ ] **Step 1: Write recovery security tests**

Cover newest valid artifact, missing files, malformed JSONL, incomplete final line, deleted model, traversal/symlink escape, wrong protocol/module/framework/project, and fallback to an older valid promotion.

- [ ] **Step 2: Write page recovery tests**

Populate project/form fields and assert inference unlocks. Change project or framework and assert it locks. Recovery must not launch a subprocess.

- [ ] **Step 3: Run RED tests**

```powershell
python -m pytest tests/training_modules/test_recovery.py tests/gui/test_training_page.py -q
```

- [ ] **Step 4: Implement a bounded journal scan**

Sort contained job directories by job/event modification time descending. Reuse strict JSON and containment rules; return the first valid matching artifact. Never scan outside the project or guess module model paths.

- [ ] **Step 5: Verify and commit**

```powershell
python -m pytest tests/training_modules/test_recovery.py tests/gui/test_training_page.py -q
git add tts_builder/training_modules/recovery.py tts_builder/gui/training_page.py tests/training_modules/test_recovery.py tests/gui/test_training_page.py
git commit -m "feat: recover promoted models for inference"
```

---

### Task 8: Documentation and dual-repository acceptance

**Repositories:** both repositories

**Files:**
- Modify: AudioMiner `README.md`
- Modify: AudioMiner `tests/integration/test_voice_pipeline_module.py`
- Modify: voice-pipeline `README.md`

**Interfaces:**
- Real handshake expects protocol v2, generic `training_data`, and `infer`.
- Documentation explains the permanent pages, promotion gate, TXT inference, and output directory.

- [ ] **Step 1: Update the opt-in integration assertion**

With both variables set, assert protocol 2, the training-data descriptor, and `infer`. Keep the literal skip reason when absent.

- [ ] **Step 2: Update both READMEs**

AudioMiner documents the GUI workflow and automatic output. voice-pipeline documents `module infer` as a machine interface while preserving standalone `infer synthesize` instructions.

- [ ] **Step 3: Run both full suites**

AudioMiner:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
$env:VOICE_PIPELINE_PYTHON='D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe'
$env:VOICE_PIPELINE_ROOT='D:\AI-Training\voice-clone\voice-pipeline\voice-pipeline'
python -m pytest -q --basetemp D:\AI-Training\voice-clone\.pytest-audioclone-v2
```

voice-pipeline:

```powershell
python -m pytest -q --basetemp D:\AI-Training\voice-clone\.pytest-voice-pipeline-v2
```

- [ ] **Step 4: Run real acceptance**

1. Verify standalone AudioMiner with no module.
2. Verify real protocol-v2 handshake and two sidebar entries.
3. Verify three permanent pages in Chinese, English, and Japanese.
4. Select the existing Acane project and recover its promoted model.
5. Run one short protocol inference and verify playback plus displayed output directory.
6. Confirm no new model weight, ZIP, EXE, or duplicated bundle.

- [ ] **Step 5: Clean only named caches**

Resolve and verify each directory is below `D:/AI-Training/voice-clone`, then remove only `.pytest-audioclone-v2` and `.pytest-voice-pipeline-v2`.

- [ ] **Step 6: Commit documentation separately**

AudioMiner:

```powershell
git add README.md tests/integration/test_voice_pipeline_module.py
git commit -m "docs: explain the protocol-v2 training workspace"
```

voice-pipeline:

```powershell
git add README.md
git commit -m "docs: document module inference protocol"
```
