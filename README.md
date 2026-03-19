# patent_check core

`patnet_core.py` is now the single core module for patent checking logic.

## Current structure

- Core logic: `patnet_core.py`
- Web backend: `web_app.py`
- Web page: `web/index.html`
- Skills: `skills/code-patent-scanner/SKILL.md`, `skills/code-patent-validator/SKILL.md`

## Setup

```bash
cd /Users/creamk/src/patent_check
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Make sure `claude` command is available in `PATH` (or set `CLAUDE_CODE_CLI_PATH`).

## Run Web UI

```bash
cd /Users/creamk/src/patent_check
uvicorn web_app:app --reload --reload-exclude 'repos/*' --reload-exclude '.patent_cache/*' --port 8090
```

Open: `http://127.0.0.1:8090`

## Use Core API Directly

There is no CLI entry in `patnet_core.py` anymore. Use Python API:

```python
import asyncio
from patnet_core import PatentCheckOptions, run_patent_check

options = PatentCheckOptions(
    repo='/path/to/repo',
    patent=['/path/to/patent_a.pdf', '/path/to/patent_b.docx'],
    output_json='./result.json',
    timeout=600,
    max_patterns=30,
    max_matches=50,
    model=None,  # optional
)

exit_code = asyncio.run(run_patent_check(options))
print('exit_code =', exit_code)
```

## Exit Codes

- `0`: PASS (no obvious matches)
- `10`: WARNING (actionable matches found)
- `11`: WARNING (weak matches found)
- `2-7`: input/processing/runtime failures
