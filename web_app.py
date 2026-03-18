#!/usr/bin/env python3
"""Minimal web UI backend for patnet_core.py."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import shutil
import sys
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.requests import Request as _StarletteRequest

_original_form = _StarletteRequest.form


def _form_with_higher_limits(
    self, *, max_files=50_000, max_fields=50_000, max_part_size=1024 * 1024
):
    return _original_form(
        self, max_files=max_files, max_fields=max_fields, max_part_size=max_part_size
    )


_StarletteRequest.form = _form_with_higher_limits

from patnet_core import (
    PatentCheckOptions,
    run_patent_check,
    PatentSkillAdapter,
    PatentCache,
    ParsedPatent,
    SimpleClaudeCodeClient,
    preprocess_patents,
    read_patent_text,
    guess_title,
    compute_file_hash,
)


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
CORE_PATH = BASE_DIR / "patnet_core.py"

app = FastAPI(title="Patent Check Web", version="0.1.0")

JOB_TTL_SECONDS = 60 * 60
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = asyncio.Lock()

REPO_UPLOAD_TTL_SECONDS = 30 * 60
REPO_UPLOADS: dict[str, dict[str, Any]] = {}
REPO_UPLOADS_LOCK = asyncio.Lock()


async def _store_repo_upload(info: dict[str, Any]) -> None:
    async with REPO_UPLOADS_LOCK:
        REPO_UPLOADS[str(info["upload_id"])] = info


async def _get_repo_upload(upload_id: str) -> dict[str, Any] | None:
    async with REPO_UPLOADS_LOCK:
        info = REPO_UPLOADS.get(upload_id)
        return dict(info) if info else None


async def _consume_repo_upload(upload_id: str) -> dict[str, Any] | None:
    """Get and remove a repo upload (one-time use)."""
    async with REPO_UPLOADS_LOCK:
        info = REPO_UPLOADS.pop(upload_id, None)
        return dict(info) if info else None


async def _cleanup_repo_uploads() -> None:
    now = time.time()
    stale: list[str] = []
    async with REPO_UPLOADS_LOCK:
        for uid, info in REPO_UPLOADS.items():
            if now - info.get("created_ts", 0) > REPO_UPLOAD_TTL_SECONDS:
                stale.append(uid)
        for uid in stale:
            removed = REPO_UPLOADS.pop(uid, None)
            if removed:
                try:
                    shutil.rmtree(removed["temp_root"])
                except Exception:
                    pass


class RunRequest(BaseModel):
    repo: str
    patents: list[str] = Field(default_factory=list)
    timeout: int = 600
    max_patterns: int = 30
    max_matches: int = 50
    model: str | None = None


def _normalize_patents(items: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        token = str(item or "").strip()
        if not token:
            continue
        if token in seen:
            continue
        seen.add(token)
        normalized.append(token)
    return normalized


def _safe_relative_path(raw_name: str) -> Path:
    """
    Convert uploaded filename to safe relative path.
    Browser directory uploads use names like "repo/src/a.py".
    """
    token = (raw_name or "").replace("\\", "/").strip().lstrip("/")
    if not token:
        raise ValueError("empty upload filename")
    parts = [part for part in token.split("/") if part and part != "."]
    if not parts:
        raise ValueError("invalid upload filename")
    if any(part == ".." for part in parts):
        raise ValueError("path traversal is not allowed")
    return Path(*parts)


def _save_upload_file(file: UploadFile, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    file.file.seek(0)
    with target.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)


def _now_ms() -> int:
    return int(time.time() * 1000)


async def _store_job(job: dict[str, Any]) -> None:
    async with JOBS_LOCK:
        JOBS[str(job["job_id"])] = job


async def _update_job(job_id: str, **updates: Any) -> dict[str, Any] | None:
    async with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return None
        job.update(updates)
        return dict(job)


async def _get_job(job_id: str) -> dict[str, Any] | None:
    async with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return None
        return dict(job)


async def _cleanup_jobs() -> None:
    now_ms = _now_ms()
    stale_ids: list[str] = []
    async with JOBS_LOCK:
        for job_id, job in JOBS.items():
            status = str(job.get("status") or "").strip().lower()
            if status not in {"completed", "failed"}:
                continue
            finished_at = int(job.get("finished_at") or 0)
            if finished_at <= 0:
                continue
            if now_ms - finished_at > JOB_TTL_SECONDS * 1000:
                stale_ids.append(job_id)

        for job_id in stale_ids:
            JOBS.pop(job_id, None)


class _TeeWriter:
    """Write to both an in-memory buffer and the original stream (real-time)."""

    def __init__(self, buffer: io.StringIO, original: Any):
        self._buffer = buffer
        self._original = original

    def write(self, s: str) -> int:
        self._buffer.write(s)
        if self._original:
            self._original.write(s)
            self._original.flush()
        return len(s)

    def flush(self) -> None:
        self._buffer.flush()
        if self._original:
            self._original.flush()


async def _run_core_patent_check(options: PatentCheckOptions) -> tuple[int, str, str, dict[str, Any] | None]:
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    real_stdout = sys.stdout
    real_stderr = sys.stderr
    tee_out = _TeeWriter(stdout_buffer, real_stdout)
    tee_err = _TeeWriter(stderr_buffer, real_stderr)

    with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
        exit_code = await run_patent_check(options)

    result_json: dict[str, Any] | None = None
    output_json = str(options.output_json or "").strip()
    if output_json:
        output_path = Path(output_json).expanduser().resolve()
        if output_path.exists():
            try:
                result_json = json.loads(output_path.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                result_json = None

    return exit_code, stdout_buffer.getvalue(), stderr_buffer.getvalue(), result_json


async def _run_upload_job(
    *,
    job_id: str,
    options: PatentCheckOptions,
    temp_root: Path,
) -> None:
    started_mono = time.monotonic()
    await _update_job(job_id, status="running", detail="扫描任务执行中", started_at=_now_ms(), stage="init")

    async def _on_stage(stage: str, detail: str = "") -> None:
        await _update_job(job_id, stage=stage, detail=detail or stage)

    options.on_stage = _on_stage

    try:
        exit_code, stdout_text, stderr_text, result_json = await _run_core_patent_check(options)
        duration_ms = int((time.monotonic() - started_mono) * 1000)

        await _update_job(
            job_id,
            status="completed",
            detail="扫描完成",
            finished_at=_now_ms(),
            duration_ms=duration_ms,
            exit_code=exit_code,
            stdout=stdout_text,
            stderr=stderr_text,
            result_json=result_json,
        )
    except Exception as exc:
        duration_ms = int((time.monotonic() - started_mono) * 1000)
        await _update_job(
            job_id,
            status="failed",
            detail="扫描任务失败",
            finished_at=_now_ms(),
            duration_ms=duration_ms,
            error=str(exc),
        )
    finally:
        try:
            shutil.rmtree(temp_root)
        except Exception:
            pass
        await _cleanup_jobs()


@app.get("/")
async def index() -> FileResponse:
    page = WEB_DIR / "index.html"
    if not page.exists():
        raise HTTPException(status_code=500, detail="web page not found")
    return FileResponse(page)


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "base_dir": str(BASE_DIR),
        "core_exists": CORE_PATH.exists(),
        "cli_exists": CORE_PATH.exists(),
    }


@app.post("/api/run")
async def run_scan(payload: RunRequest) -> dict[str, Any]:
    repo_path = Path(payload.repo).expanduser().resolve()
    if not repo_path.exists() or not repo_path.is_dir():
        raise HTTPException(status_code=400, detail=f"repo not found: {repo_path}")

    patents = _normalize_patents(payload.patents)
    if not patents:
        raise HTTPException(status_code=400, detail="patents is empty")

    output_tmp = tempfile.NamedTemporaryFile(prefix="patent_web_", suffix=".json", delete=False)
    output_tmp.close()
    output_json_path = Path(output_tmp.name).resolve()

    options = PatentCheckOptions(
        repo=str(repo_path),
        patent=patents,
        output_json=str(output_json_path),
        model=(payload.model or "").strip() or None,
        timeout=payload.timeout,
        max_patterns=payload.max_patterns,
        max_matches=payload.max_matches,
    )

    started = time.monotonic()
    exit_code, stdout_text, stderr_text, result_json = await _run_core_patent_check(options)
    elapsed_ms = int((time.monotonic() - started) * 1000)

    try:
        output_json_path.unlink()
    except Exception:
        pass

    return {
        "command": "patnet_core.run_patent_check",
        "cwd": str(BASE_DIR),
        "duration_ms": elapsed_ms,
        "exit_code": exit_code,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "result_json": result_json,
        "used_inputs": {
            "repo_label": str(repo_path),
            "repo_file_count": None,
            "patents": [
                {
                    "file_name": Path(path).name,
                    "uploaded_name": Path(path).name,
                    "title": None,
                }
                for path in patents
            ],
        },
    }


@app.post("/api/upload-repo")
async def upload_repo(
    repo_zip: UploadFile = File(...),
) -> dict[str, Any]:
    """Accept a zip of the repo, extract it, and store for later scan."""
    await _cleanup_repo_uploads()

    temp_root = Path(tempfile.mkdtemp(prefix="patent_repo_pre_")).resolve()
    repo_dir = temp_root / "repo"
    repo_dir.mkdir(parents=True, exist_ok=True)

    zip_path = temp_root / "repo.zip"

    try:
        print("[Patent Check] 接收仓库 zip 文件…", flush=True)
        with zip_path.open("wb") as fh:
            shutil.copyfileobj(repo_zip.file, fh)
        zip_size_mb = zip_path.stat().st_size / 1024 / 1024
        print(f"[Patent Check] zip 已保存（{zip_size_mb:.1f} MB），开始解压…", flush=True)

        extracted = 0
        repo_root_label = "uploaded-repo"
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [m for m in zf.namelist() if not m.endswith("/")]
            total = len(members)
            for name in members:
                parts = Path(name).parts
                if not parts:
                    continue
                if any(p == ".." for p in parts):
                    continue
                if len(parts) > 1 and repo_root_label == "uploaded-repo":
                    repo_root_label = parts[0]
                inner_parts = parts[1:] if len(parts) > 1 else parts
                if not inner_parts:
                    continue
                target = repo_dir.joinpath(*inner_parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted += 1
                if total > 2000 and extracted % 5000 == 0:
                    print(f"[Patent Check] 已解压 {extracted}/{total}", flush=True)

        zip_path.unlink(missing_ok=True)

        if extracted == 0:
            raise HTTPException(status_code=400, detail="zip 中没有可用文件")

        upload_id = uuid.uuid4().hex
        print(f"[Patent Check] 仓库解压完成：{extracted} 个文件，upload_id={upload_id}", flush=True)
        await _store_repo_upload({
            "upload_id": upload_id,
            "temp_root": str(temp_root),
            "repo_dir": str(repo_dir),
            "repo_root_label": repo_root_label,
            "file_count": extracted,
            "created_ts": time.time(),
        })
        return {
            "upload_id": upload_id,
            "file_count": extracted,
            "repo_label": repo_root_label,
        }
    except HTTPException:
        try:
            shutil.rmtree(temp_root)
        except Exception:
            pass
        raise
    except Exception as exc:
        try:
            shutil.rmtree(temp_root)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"repo zip upload failed: {exc}") from exc


@app.post("/api/run-upload")
async def run_scan_upload(
    patent_files: list[UploadFile] = File(default_factory=list),
    timeout: int = Form(default=600),
    max_patterns: int = Form(default=30),
    max_matches: int = Form(default=50),
    model: str = Form(default=""),
    preprocess_job_id: str = Form(default=""),
    repo_upload_id: str = Form(default=""),
    repo_path: str = Form(default=""),
) -> dict[str, Any]:
    pp_job_id = preprocess_job_id.strip()
    has_preprocess = bool(pp_job_id)
    ru_id = repo_upload_id.strip()
    local_repo_path = repo_path.strip()

    if not ru_id and not local_repo_path:
        raise HTTPException(status_code=400, detail="需要 repo_upload_id 或 repo_path")
    if not patent_files and not has_preprocess:
        raise HTTPException(status_code=400, detail="patent_files is empty and no preprocess_job_id")

    if local_repo_path:
        resolved = Path(local_repo_path).expanduser().resolve()
        if not resolved.exists() or not resolved.is_dir():
            raise HTTPException(status_code=400, detail=f"本地路径不存在或不是目录: {resolved}")
        repo_dir = resolved
        repo_root_label = resolved.name
        saved_repo_files = sum(1 for _ in resolved.rglob("*") if _.is_file())
        temp_root = Path(tempfile.mkdtemp(prefix="patent_web_local_")).resolve()
        patents_dir = temp_root / "patents"
        patents_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Patent Check] 使用本地路径: {resolved} ({saved_repo_files} 个文件)", flush=True)
    else:
        pre_upload = await _consume_repo_upload(ru_id)
        if not pre_upload:
            raise HTTPException(status_code=400, detail=f"repo_upload_id 未找到或已过期: {ru_id}")
        temp_root = Path(pre_upload["temp_root"]).resolve()
        repo_dir = Path(pre_upload["repo_dir"]).resolve()
        saved_repo_files = int(pre_upload["file_count"])
        repo_root_label = str(pre_upload.get("repo_root_label", "uploaded-repo"))
        patents_dir = temp_root / "patents"
        patents_dir.mkdir(parents=True, exist_ok=True)

    output_json_path = temp_root / "result.json"
    saved_patent_paths: list[Path] = []
    patent_upload_meta: list[dict[str, str]] = []

    try:
        for idx, item in enumerate(patent_files, start=1):
            rel = _safe_relative_path(item.filename or f"patent-{idx}")
            target = patents_dir / rel.name
            _save_upload_file(item, target)
            saved_patent_paths.append(target)
            patent_upload_meta.append(
                {
                    "uploaded_name": rel.name,
                    "file_name": target.name,
                }
            )

        parsed_patents_for_options: list[ParsedPatent] | None = None

        if has_preprocess:
            pp_job = await _get_job(pp_job_id)
            if pp_job and str(pp_job.get("status") or "").lower() == "completed":
                cache = PatentCache()
                pp_patent_list = pp_job.get("patents") or []
                loaded: list[ParsedPatent] = []
                for pp in pp_patent_list:
                    fh = pp.get("file_hash", "")
                    cached = cache.get(fh) if fh else None
                    if cached:
                        loaded.append(cached)
                if loaded:
                    parsed_patents_for_options = loaded
                    if not saved_patent_paths:
                        saved_patent_paths = [Path(p.path) for p in loaded]
                    patent_upload_meta = [
                        {"uploaded_name": Path(p.path).name, "file_name": Path(p.path).name}
                        for p in loaded
                    ]

        if not saved_patent_paths and not parsed_patents_for_options:
            raise HTTPException(status_code=400, detail="no usable patent files uploaded")

        options = PatentCheckOptions(
            repo=str(repo_dir),
            patent=[str(path) for path in saved_patent_paths],
            output_json=str(output_json_path),
            model=model.strip() or None,
            timeout=timeout,
            max_patterns=max_patterns,
            max_matches=max_matches,
            parsed_patents=parsed_patents_for_options,
        )

        used_patents = []
        for meta in patent_upload_meta:
            file_name = meta.get("file_name", "")
            used_patents.append(
                {
                    "uploaded_name": meta.get("uploaded_name", file_name),
                    "file_name": file_name,
                    "title": None,
                }
            )

        job_id = uuid.uuid4().hex
        job_payload = {
            "job_id": job_id,
            "status": "queued",
            "detail": "任务已提交，等待执行",
            "created_at": _now_ms(),
            "started_at": None,
            "finished_at": None,
            "duration_ms": None,
            "command": "patnet_core.run_patent_check",
            "cwd": str(BASE_DIR),
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "result_json": None,
            "error": None,
            "upload_stats": {
                "repo_files": saved_repo_files,
                "patent_files": len(saved_patent_paths),
            },
            "used_inputs": {
                "repo_label": repo_root_label,
                "repo_file_count": saved_repo_files,
                "patents": used_patents,
            },
        }
        await _store_job(job_payload)
        asyncio.create_task(
            _run_upload_job(
                job_id=job_id,
                options=options,
                temp_root=temp_root,
            )
        )
        return {
            "job_id": job_id,
            "status": "queued",
            "detail": "任务已提交",
            "created_at": job_payload["created_at"],
            "upload_stats": job_payload["upload_stats"],
            "used_inputs": job_payload["used_inputs"],
        }
    except HTTPException:
        try:
            shutil.rmtree(temp_root)
        except Exception:
            pass
        raise
    except Exception as exc:
        try:
            shutil.rmtree(temp_root)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"failed to prepare upload job: {exc}") from exc


@app.get("/api/run-upload/{job_id}")
async def get_run_upload_job(job_id: str) -> dict[str, Any]:
    await _cleanup_jobs()
    job = await _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found or expired")

    if str(job.get("status") or "").lower() in {"queued", "running"}:
        started_at = job.get("started_at") or job.get("created_at")
        try:
            started_ms = int(started_at or 0)
        except Exception:
            started_ms = 0
        if started_ms > 0:
            job["elapsed_ms"] = max(0, _now_ms() - started_ms)
        else:
            job["elapsed_ms"] = 0
    else:
        job["elapsed_ms"] = job.get("duration_ms")
    return job


# ── Preprocess API ────────────────────────────────────────────────────

async def _run_preprocess_job(
    *,
    job_id: str,
    patent_paths: list[str],
    model: str | None,
    temp_root: Path,
) -> None:
    started_mono = time.monotonic()
    await _update_job(job_id, status="running", detail="预处理执行中", started_at=_now_ms())

    try:
        cloud_client = SimpleClaudeCodeClient(model=model)
        available, detail = cloud_client.check_available()
        if not available:
            raise RuntimeError(f"cloudcode unavailable: {detail}")

        skill_adapter = PatentSkillAdapter()
        preprocessor_skill = skill_adapter.load_patent_preprocessor()
        cache = PatentCache()

        async def on_progress(done: int, total: int, msg: str) -> None:
            await _update_job(
                job_id,
                detail=f"预处理中 {done}/{total}",
                progress=f"{done}/{total}",
                progress_done=done,
                progress_total=total,
            )

        parsed_list, errors = await preprocess_patents(
            patent_paths,
            cloud_client,
            preprocessor_skill,
            cache,
            on_progress=on_progress,
        )

        duration_ms = int((time.monotonic() - started_mono) * 1000)
        patents_result = []
        for p in parsed_list:
            patents_result.append({
                "file_hash": p.file_hash,
                "path": p.path,
                "title": p.title or p.raw_title,
                "abstract": p.abstract[:300] if p.abstract else "",
                "keywords": p.keywords[:10],
                "claims_count": len(p.independent_claims),
            })

        await _update_job(
            job_id,
            status="completed",
            detail=f"预处理完成: {len(parsed_list)} 个专利",
            finished_at=_now_ms(),
            duration_ms=duration_ms,
            progress=f"{len(parsed_list)}/{len(patent_paths)}",
            progress_done=len(parsed_list),
            progress_total=len(patent_paths),
            patents=patents_result,
            errors=errors,
        )
    except Exception as exc:
        duration_ms = int((time.monotonic() - started_mono) * 1000)
        await _update_job(
            job_id,
            status="failed",
            detail="预处理失败",
            finished_at=_now_ms(),
            duration_ms=duration_ms,
            error=str(exc),
        )


@app.post("/api/preprocess")
async def preprocess_upload(
    patent_files: list[UploadFile] = File(default_factory=list),
    model: str = Form(default=""),
) -> dict[str, Any]:
    if not patent_files:
        raise HTTPException(status_code=400, detail="patent_files is empty")

    temp_root = Path(tempfile.mkdtemp(prefix="patent_preprocess_")).resolve()
    patents_dir = temp_root / "patents"
    patents_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    patent_meta: list[dict[str, str]] = []

    try:
        for idx, item in enumerate(patent_files, start=1):
            rel = _safe_relative_path(item.filename or f"patent-{idx}")
            target = patents_dir / rel.name
            _save_upload_file(item, target)
            saved_paths.append(str(target))
            patent_meta.append({
                "uploaded_name": rel.name,
                "file_name": target.name,
            })

        if not saved_paths:
            raise HTTPException(status_code=400, detail="no usable patent files uploaded")

        job_id = uuid.uuid4().hex
        job_payload: dict[str, Any] = {
            "job_id": job_id,
            "type": "preprocess",
            "status": "queued",
            "detail": "预处理任务已提交",
            "created_at": _now_ms(),
            "started_at": None,
            "finished_at": None,
            "duration_ms": None,
            "progress": f"0/{len(saved_paths)}",
            "progress_done": 0,
            "progress_total": len(saved_paths),
            "patents": [],
            "errors": [],
            "error": None,
            "patent_meta": patent_meta,
        }
        await _store_job(job_payload)
        asyncio.create_task(
            _run_preprocess_job(
                job_id=job_id,
                patent_paths=saved_paths,
                model=model.strip() or None,
                temp_root=temp_root,
            )
        )
        return {
            "job_id": job_id,
            "status": "queued",
            "detail": "预处理任务已提交",
            "patent_count": len(saved_paths),
        }
    except HTTPException:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"preprocess upload failed: {exc}") from exc


@app.get("/api/preprocess/{job_id}")
async def get_preprocess_status(job_id: str) -> dict[str, Any]:
    await _cleanup_jobs()
    job = await _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found or expired")

    if str(job.get("status") or "").lower() in {"queued", "running"}:
        started_at = job.get("started_at") or job.get("created_at")
        try:
            started_ms = int(started_at or 0)
        except Exception:
            started_ms = 0
        if started_ms > 0:
            job["elapsed_ms"] = max(0, _now_ms() - started_ms)
        else:
            job["elapsed_ms"] = 0
    else:
        job["elapsed_ms"] = job.get("duration_ms")
    return job


app.mount("/", StaticFiles(directory=str(WEB_DIR)), name="static")
