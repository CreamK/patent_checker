#!/usr/bin/env python3
"""Core patent checker logic extracted from oscp-agents patent_check agent.

The core flow:
1. Read patent files.
2. Ask cloudcode (Claude Code SDK) to read repository and extract technical patterns.
3. Match extracted patterns with uploaded patent texts.
4. Normalize and summarize match results.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from shutil import which
from typing import Any, Callable
from xml.etree import ElementTree


MAX_PATENT_FILES = 2000
MAX_PATENT_TEXT_CHARS = 12000
MAX_PATTERNS = 30
MAX_MATCHES = 50
PATENT_HIT_SIMILARITY_THRESHOLD = 0.5
DEFAULT_RECALL_TOP_K = 40
DEFAULT_RERANK_TOP_N = 15
DEFAULT_DEEP_MAX_CONCURRENCY = 3
PREPROCESS_BATCH_SIZE = 3
IDLE_TIMEOUT = 300
MAX_NUDGES = 2
MAX_FULL_RETRIES = 1
RETRY_DELAY = 10
NUDGE_MESSAGES = [
    "You appear to be stuck or still processing. Stop all tool usage now and immediately "
    "output your final JSON result based on what you have analyzed so far. "
    "Output ONLY the JSON object, no markdown, no explanation.",
    "STOP. Output the JSON result NOW with whatever patterns you have found so far. "
    "Do not use any more tools. Respond with ONLY the raw JSON object.",
]


@dataclass
class SkillLoadResult:
    name: str
    loaded: bool
    source: str
    content: str
    note: str = ""


@dataclass
class PatentCheckOptions:
    repo: str
    patent: list[str]
    output_json: str | None = None
    skills_root: str | None = None
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    timeout: int = 600
    max_patent_files: int = MAX_PATENT_FILES
    max_patent_text_chars: int = MAX_PATENT_TEXT_CHARS
    max_patterns: int = MAX_PATTERNS
    max_matches: int = MAX_MATCHES
    recall_top_k: int = DEFAULT_RECALL_TOP_K
    rerank_top_n: int = DEFAULT_RERANK_TOP_N
    deep_max_concurrency: int = DEFAULT_DEEP_MAX_CONCURRENCY
    cache_dir: str | None = None
    preprocess_job_id: str | None = None
    parsed_patents: list[Any] | None = None
    on_stage: Callable | None = None
    idle_timeout: int = IDLE_TIMEOUT


@dataclass
class ParsedPatent:
    file_hash: str
    path: str
    raw_title: str
    title: str
    abstract: str
    independent_claims: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    full_text: str = ""
    summary_text: str = ""

    def build_summary_text(self) -> str:
        parts = []
        if self.abstract:
            parts.append(self.abstract)
        for claim in self.independent_claims[:2]:
            if claim:
                parts.append(claim)
        return "\n\n".join(parts) or self.full_text[:800]

    def to_legacy_doc(self) -> dict[str, str]:
        return {
            "id": f"patent-{self.file_hash[:8]}",
            "path": self.path,
            "title": self.title or self.raw_title,
            "text": trim_text(self.full_text, MAX_PATENT_TEXT_CHARS),
        }


class PatentCache:
    """SHA256-based disk cache for parsed patent metadata."""

    def __init__(self, cache_dir: str | Path | None = None):
        if cache_dir:
            self.cache_dir = Path(cache_dir).expanduser().resolve()
        else:
            self.cache_dir = Path(__file__).resolve().parent / ".patent_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, file_hash: str) -> Path:
        return self.cache_dir / f"{file_hash}.json"

    def has(self, file_hash: str) -> bool:
        return self._path_for(file_hash).exists()

    def get(self, file_hash: str) -> ParsedPatent | None:
        path = self._path_for(file_hash)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            return ParsedPatent(
                file_hash=data.get("file_hash", file_hash),
                path=data.get("path", ""),
                raw_title=data.get("raw_title", ""),
                title=data.get("title", ""),
                abstract=data.get("abstract", ""),
                independent_claims=data.get("independent_claims", []),
                keywords=data.get("keywords", []),
                full_text=data.get("full_text", ""),
                summary_text=data.get("summary_text", ""),
            )
        except Exception:
            return None

    def put(self, parsed: ParsedPatent) -> None:
        path = self._path_for(parsed.file_hash)
        try:
            path.write_text(
                json.dumps(asdict(parsed), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass


def compute_file_hash(file_path: Path) -> str:
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class PatentSkillAdapter:
    """Load scanner/validator skill markdown from local files with fallback."""

    DEFAULT_SKILL_ROOT = Path(__file__).resolve().parent / "skills"
    SCANNER_SKILL_REL = Path("code-patent-scanner") / "SKILL.md"
    VALIDATOR_SKILL_REL = Path("code-patent-validator") / "SKILL.md"
    PREPROCESSOR_SKILL_REL = Path("patent-preprocessor") / "SKILL.md"
    MAX_SKILL_CHARS = 12000

    SCANNER_FALLBACK = """# Code Patent Scanner (Fallback)
Role: discover distinctive technical patterns from code.
Rules:
1. Autonomously explore the repository and analyze important source files.
2. Score each pattern by 4 dimensions:
   - distinctiveness (0-4)
   - sophistication (0-3)
   - system_impact (0-3)
   - frame_shift (0-3)
3. Report ALL patterns found regardless of score. The caller will rank and filter.
4. For each pattern provide: title, category, source_files, why_distinctive,
   claim_angles (method/system/apparatus), abstract_mechanism, concrete_reference.
5. Output must be strict JSON object.
"""

    VALIDATOR_FALLBACK = """# Code Patent Validator (Fallback)
Role: validate scanner findings against uploaded patent documents.
Rules:
1. Compare each code pattern with each uploaded patent text.
2. Return match_level: high|medium|low|none and similarity 0-1.
3. Provide concise reason and evidence snippets.
4. Output must be strict JSON object.
"""

    PREPROCESSOR_FALLBACK = """# Patent Preprocessor (Fallback)
Role: Extract structured metadata from patent documents.
Rules:
1. For each patent text, extract: title, abstract (200-500 chars), independent claims, keywords.
2. Independent claims do not reference other claims.
3. Keywords: 5-15 key technical terms.
4. Output must be strict JSON object with "patents" array.
"""

    def __init__(self, skills_root: str | Path | None = None):
        env_root = (
            (os.getenv("PATNET_CORE_SKILLS_DIR") or "").strip()
            or (os.getenv("PATENT_CLI_SKILLS_DIR") or "").strip()
        )
        if skills_root:
            self.skills_root = Path(skills_root).expanduser().resolve()
        elif env_root:
            self.skills_root = Path(env_root).expanduser().resolve()
        else:
            self.skills_root = self.DEFAULT_SKILL_ROOT

    def load_code_patent_scanner(self) -> SkillLoadResult:
        return self._load_skill(
            name="code-patent-scanner",
            relative_path=self.SCANNER_SKILL_REL,
            fallback=self.SCANNER_FALLBACK,
        )

    def load_code_patent_validator(self) -> SkillLoadResult:
        return self._load_skill(
            name="code-patent-validator",
            relative_path=self.VALIDATOR_SKILL_REL,
            fallback=self.VALIDATOR_FALLBACK,
        )

    def load_patent_preprocessor(self) -> SkillLoadResult:
        return self._load_skill(
            name="patent-preprocessor",
            relative_path=self.PREPROCESSOR_SKILL_REL,
            fallback=self.PREPROCESSOR_FALLBACK,
        )

    def _load_skill(self, name: str, relative_path: Path, fallback: str) -> SkillLoadResult:
        path = (self.skills_root / relative_path).resolve()
        if path.exists() and path.is_file():
            try:
                text = path.read_text(encoding="utf-8", errors="ignore").strip()
                if text:
                    return SkillLoadResult(
                        name=name,
                        loaded=True,
                        source=str(path),
                        content=self._trim(text),
                    )
                return SkillLoadResult(
                    name=name,
                    loaded=False,
                    source=str(path),
                    content=self._trim(fallback),
                    note="skill file is empty; fallback used",
                )
            except Exception as exc:
                return SkillLoadResult(
                    name=name,
                    loaded=False,
                    source=str(path),
                    content=self._trim(fallback),
                    note=f"failed reading skill: {exc}",
                )

        return SkillLoadResult(
            name=name,
            loaded=False,
            source=str(path),
            content=self._trim(fallback),
            note="skill file not found; fallback used",
        )

    def _trim(self, text: str) -> str:
        normalized = text.replace("\r\n", "\n").strip()
        if len(normalized) <= self.MAX_SKILL_CHARS:
            return normalized
        return normalized[: self.MAX_SKILL_CHARS] + "\n...[skill content truncated]..."


class SimpleClaudeCodeClient:
    """Minimal Claude Code SDK wrapper for this standalone core module."""

    def __init__(
        self,
        model: str | None = None,
        anthropic_api_key: str | None = None,
        anthropic_base_url: str | None = None,
    ):
        self.model = model
        self.anthropic_api_key = anthropic_api_key or (os.getenv("ANTHROPIC_API_KEY") or "").strip()
        self.anthropic_base_url = anthropic_base_url or (os.getenv("ANTHROPIC_BASE_URL") or "").strip()

    @staticmethod
    def _is_windows() -> bool:
        return sys.platform == "win32"

    def _resolve_claude_cli_path(self) -> str | None:
        configured = (os.getenv("CLAUDE_CODE_CLI_PATH") or "").strip().strip('"').strip("'")
        if configured:
            return configured

        # On Windows, prefer .exe (which anyio.open_process can exec directly)
        # over .cmd (which needs cmd.exe /c and may fail with SDK's subprocess)
        candidates = ["claude"]
        if self._is_windows():
            candidates = ["claude.exe", "claude.cmd", "claude"]

        for candidate in candidates:
            resolved = which(candidate)
            if resolved:
                return resolved

        # Windows-specific fallback paths
        if self._is_windows():
            home = Path.home()
            appdata = os.getenv("APPDATA", "")
            localappdata = os.getenv("LOCALAPPDATA", "")
            win_paths = [
                Path(localappdata) / "Programs" / "claude" / "claude.exe" if localappdata else None,
                Path(appdata) / "npm" / "claude.cmd" if appdata else None,
                home / "AppData" / "Roaming" / "npm" / "claude.cmd",
                home / "node_modules" / ".bin" / "claude.cmd",
                home / ".claude" / "local" / "claude.exe",
            ]
            for p in win_paths:
                if p and p.exists() and p.is_file():
                    return str(p)

        return None

    @staticmethod
    def _build_subprocess_cmd(cli_path: str, args: list[str]) -> list[str]:
        """Wrap .cmd/.bat in cmd.exe /c on Windows for proper execution."""
        if sys.platform == "win32" and cli_path.lower().endswith((".cmd", ".bat")):
            return ["cmd.exe", "/c", cli_path] + args
        return [cli_path] + args

    @staticmethod
    def _resolve_cmd_to_exe(cmd_path: str) -> str | None:
        """On Windows, .cmd files can't be exec'd by anyio.open_process.
        Try to find a real .exe alternative."""
        try:
            # 1. Check .exe sibling in same directory (e.g. claude.exe next to claude.cmd)
            exe_sibling = Path(cmd_path).with_suffix(".exe")
            if exe_sibling.exists():
                return str(exe_sibling)

            # 2. Check SDK bundled binary
            try:
                import claude_agent_sdk
                bundled = Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude.exe"
                if bundled.exists():
                    return str(bundled)
            except Exception:
                pass

        except Exception:
            pass
        return None

    @staticmethod
    def _make_windows_cmd_transport(options: Any, cmd_cli_path: str) -> Any:
        """Create a custom transport that wraps .cmd with cmd.exe /c on Windows.
        Falls back to None if SDK internals aren't accessible."""
        try:
            from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

            class _WindowsCmdTransport(SubprocessCLITransport):
                """Override _build_command to prepend cmd.exe /c for .cmd files."""

                def __init__(self, prompt: Any, options: Any, cmd_path: str):
                    self._cmd_path = cmd_path
                    super().__init__(prompt=prompt, options=options)
                    self._cli_path = cmd_path

                def _build_command(self) -> list[str]:
                    cmd = super()._build_command()
                    if cmd and cmd[0].lower().endswith((".cmd", ".bat")):
                        cmd = ["cmd.exe", "/c"] + cmd
                    return cmd

            async def _empty_stream():
                return
                yield {}  # type: ignore[unreachable]

            transport = _WindowsCmdTransport(
                prompt=_empty_stream(),
                options=options,
                cmd_path=cmd_cli_path,
            )
            print(f"[claude] Windows: using cmd.exe /c transport for {cmd_cli_path}")
            return transport
        except Exception as exc:
            print(f"[warn] Windows: failed to create cmd transport: {exc}")
            return None

    def check_available(self) -> tuple[bool, str]:
        cli_path = self._resolve_claude_cli_path()
        if not cli_path:
            return False, "claude CLI not found in PATH and CLAUDE_CODE_CLI_PATH not set"
        try:
            cmd = self._build_subprocess_cmd(cli_path, ["--version"])
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception as exc:
            return False, f"failed to run `{cli_path} --version`: {exc}"

        if completed.returncode == 0:
            version = (completed.stdout or "").strip() or "unknown"
            return True, f"{cli_path} ({version})"

        stderr_text = (completed.stderr or "").strip()
        stdout_text = (completed.stdout or "").strip()
        detail = stderr_text or stdout_text or "unknown error"
        return False, f"claude --version failed: {detail}"

    async def analyze(
        self,
        *,
        message: str,
        system_prompt: str | None,
        workspace_path: Path,
        timeout: int,
        idle_timeout: int = IDLE_TIMEOUT,
    ) -> str:
        sdk = self._load_sdk()
        session_id = str(uuid.uuid4())
        t_global_start = time.monotonic()

        print(f"[claude] session {session_id[:8]} start | model={self.model or 'default'} "
              f"| cwd={workspace_path} | timeout={timeout}s idle={idle_timeout}s")
        print(f"[claude] prompt length: {len(message)} chars")

        all_parts: list[str] = []
        total_msg_count = 0

        try:
            async with asyncio.timeout(timeout):
                for full_attempt in range(1 + MAX_FULL_RETRIES):
                    sdk_client = self._create_sdk_client(
                        sdk, system_prompt=system_prompt,
                        workspace_path=workspace_path,
                    )
                    attempt_session_id = session_id if full_attempt == 0 else str(uuid.uuid4())
                    try:
                        await sdk_client.connect()
                        if full_attempt > 0:
                            print(f"[claude] full retry #{full_attempt} | "
                                  f"new session {attempt_session_id[:8]}")

                        # Phase 1: send original prompt and receive with idle timeout
                        print(f"[claude] connected, sending query ...")
                        await sdk_client.query(message, session_id=attempt_session_id)
                        parts, msg_count, completed = await self._receive_with_idle_timeout(
                            sdk_client, idle_timeout, sdk, t_global_start,
                        )
                        all_parts.extend(parts)
                        total_msg_count += msg_count

                        if completed:
                            output = self._join_parts(all_parts)
                            if output:
                                print(f"[claude] final output: {len(output)} chars")
                                return output

                        # Phase 2: nudge in the same session to get Claude to continue
                        for nudge_idx in range(MAX_NUDGES):
                            nudge_msg = NUDGE_MESSAGES[nudge_idx % len(NUDGE_MESSAGES)]
                            print(f"[claude] nudge #{nudge_idx + 1}/{MAX_NUDGES} "
                                  f"in session {attempt_session_id[:8]}")
                            await sdk_client.query(nudge_msg, session_id=attempt_session_id)
                            nudge_parts, n_msgs, completed = await self._receive_with_idle_timeout(
                                sdk_client, idle_timeout, sdk, t_global_start,
                            )
                            all_parts.extend(nudge_parts)
                            total_msg_count += n_msgs

                            output = self._join_parts(all_parts)
                            if completed or output:
                                if output:
                                    print(f"[claude] nudge succeeded, output: {len(output)} chars")
                                    return output

                        # Phase 3: fall back to partial results from what we already have
                        partial = self._join_parts(all_parts)
                        if partial:
                            print(f"[claude] using partial result after nudges ({len(partial)} chars)")
                            return partial

                        print(f"[claude] all {MAX_NUDGES} nudges exhausted, no usable output")

                    finally:
                        elapsed = time.monotonic() - t_global_start
                        print(f"[claude] session {attempt_session_id[:8]} closing | "
                              f"{elapsed:.1f}s | msgs={total_msg_count} | parts={len(all_parts)}")
                        await self._safe_disconnect(sdk_client)

                    # Phase 4: prepare for full retry with new session
                    if full_attempt < MAX_FULL_RETRIES:
                        print(f"[claude] full retry in {RETRY_DELAY}s ...")
                        await asyncio.sleep(RETRY_DELAY)
                        all_parts.clear()
                        total_msg_count = 0

        except asyncio.TimeoutError:
            # Global timeout hit — still try to salvage partial results
            output = self._join_parts(all_parts)
            if output:
                print(f"[claude] global timeout but salvaged partial result ({len(output)} chars)")
                return output
            raise RuntimeError(
                f"claude code global timeout ({timeout}s) with no usable output"
            )

        raise RuntimeError("all attempts exhausted, no output from claude code")

    async def _receive_with_idle_timeout(
        self,
        sdk_client: Any,
        idle_timeout: int,
        sdk: dict[str, Any],
        t_global_start: float,
    ) -> tuple[list[str], int, bool]:
        """Receive SDK messages with per-message idle timeout.
        Returns (parts, msg_count, completed).
        `completed=True` means the response ended normally (ResultMessage received).
        `completed=False` means idle timeout fired — parts may still contain useful data.
        """
        AssistantMessage = sdk["AssistantMessage"]
        ResultMessage = sdk["ResultMessage"]
        TextBlock = sdk["TextBlock"]

        parts: list[str] = []
        msg_count = 0
        response_iter = sdk_client.receive_response().__aiter__()

        while True:
            try:
                sdk_message = await asyncio.wait_for(
                    response_iter.__anext__(), timeout=idle_timeout,
                )
            except StopAsyncIteration:
                return parts, msg_count, True
            except asyncio.TimeoutError:
                elapsed = time.monotonic() - t_global_start
                print(f"[claude] idle timeout ({idle_timeout}s no new message) "
                      f"| elapsed={elapsed:.1f}s msgs={msg_count}")
                return parts, msg_count, False

            msg_count += 1
            elapsed = time.monotonic() - t_global_start
            msg_type = type(sdk_message).__name__

            if isinstance(sdk_message, AssistantMessage):
                text_blocks: list[str] = []
                tool_blocks: list[str] = []
                for block in sdk_message.content:
                    if isinstance(block, TextBlock) and (block.text or "").strip():
                        parts.append(block.text)
                        text_blocks.append(block.text)
                    else:
                        tool_blocks.append(type(block).__name__)
                if text_blocks:
                    preview = text_blocks[0][:120].replace("\n", " ")
                    print(f"[claude] [{elapsed:6.1f}s] msg#{msg_count} AssistantMessage "
                          f"text={len(text_blocks)} blocks | preview: {preview}")
                if tool_blocks:
                    print(f"[claude] [{elapsed:6.1f}s] msg#{msg_count} AssistantMessage "
                          f"tool_use: {', '.join(tool_blocks)}")

            elif isinstance(sdk_message, ResultMessage):
                result_text = (sdk_message.result or "").strip()
                if result_text:
                    parts.append(result_text)
                preview = result_text[:120].replace("\n", " ") if result_text else "(empty)"
                print(f"[claude] [{elapsed:6.1f}s] msg#{msg_count} ResultMessage | preview: {preview}")

            else:
                print(f"[claude] [{elapsed:6.1f}s] msg#{msg_count} {msg_type}")

    @staticmethod
    def _join_parts(parts: list[str]) -> str:
        return "\n".join(p for p in parts if p.strip()).strip()

    async def _safe_disconnect(self, sdk_client: Any) -> None:
        if not sdk_client:
            return
        try:
            async with asyncio.timeout(5):
                await sdk_client.disconnect()
        except Exception:
            try:
                transport = getattr(sdk_client, "_transport", None)
                proc = getattr(transport, "_process", None) if transport else None
                if proc and hasattr(proc, "kill"):
                    proc.kill()
                    print("[claude] force killed stuck subprocess")
            except Exception:
                pass

    def _create_sdk_client(
        self,
        sdk: dict[str, Any],
        *,
        system_prompt: str | None,
        workspace_path: Path,
    ) -> Any:
        """Build and return a new ClaudeSDKClient (not yet connected)."""
        ClaudeAgentOptions = sdk["ClaudeAgentOptions"]
        ClaudeSDKClient = sdk["ClaudeSDKClient"]

        cli_path = self._resolve_claude_cli_path()
        if not cli_path:
            raise RuntimeError("claude CLI not found")

        env: dict[str, str] = {}
        if self.anthropic_api_key and not self.anthropic_api_key.startswith("sk-ant-xxx"):
            env["ANTHROPIC_API_KEY"] = self.anthropic_api_key
        if self.anthropic_base_url:
            env["ANTHROPIC_BASE_URL"] = self.anthropic_base_url

        effective_cli_path: str | None = cli_path
        use_cmd_transport = False

        if self._is_windows() and cli_path.lower().endswith((".cmd", ".bat")):
            resolved_exe = self._resolve_cmd_to_exe(cli_path)
            if resolved_exe:
                print(f"[claude] Windows: resolved .cmd -> .exe: {resolved_exe}")
                effective_cli_path = resolved_exe
            else:
                print(f"[claude] Windows: .cmd found ({cli_path}), "
                      f"will try SDK bundled binary or cmd.exe /c fallback")
                effective_cli_path = None
                use_cmd_transport = True

        permission_mode = "bypassPermissions"
        if os.getuid() == 0 if hasattr(os, "getuid") else False:
            permission_mode = "acceptEdits"
            print("[claude] running as root, using acceptEdits permission mode")

        opts_kwargs: dict[str, Any] = {
            "permission_mode": permission_mode,
            "cwd": str(workspace_path.resolve()),
            "model": self.model,
            "system_prompt": system_prompt,
            "env": env,
            "setting_sources": ["user", "project"],
        }
        if effective_cli_path:
            opts_kwargs["cli_path"] = effective_cli_path

        options = ClaudeAgentOptions(**opts_kwargs)

        custom_transport = None
        if use_cmd_transport and effective_cli_path is None:
            custom_transport = self._make_windows_cmd_transport(options, cli_path)

        return ClaudeSDKClient(options=options, transport=custom_transport)

    def _load_sdk(self) -> dict[str, Any]:
        try:
            from claude_agent_sdk import (
                AssistantMessage,
                ClaudeAgentOptions,
                ClaudeSDKClient,
                ResultMessage,
                TextBlock,
            )
        except ImportError as exc:
            raise RuntimeError(
                "missing dependency `claude-agent-sdk`, run `pip install -r requirements.txt`"
            ) from exc

        return {
            "AssistantMessage": AssistantMessage,
            "ClaudeAgentOptions": ClaudeAgentOptions,
            "ClaudeSDKClient": ClaudeSDKClient,
            "ResultMessage": ResultMessage,
            "TextBlock": TextBlock,
        }


def normalize_patent_paths(value: list[str]) -> list[str]:
    items: list[str] = []
    for entry in value:
        if not isinstance(entry, str):
            continue
        items.extend(re.split(r"[,\n]", entry))

    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        token = item.strip()
        if not token:
            continue
        if token in seen:
            continue
        seen.add(token)
        normalized.append(token)
    return normalized


def load_patent_documents(
    patent_paths: list[str],
    *,
    max_patent_files: int = MAX_PATENT_FILES,
    max_patent_text_chars: int = MAX_PATENT_TEXT_CHARS,
) -> tuple[list[dict[str, str]], list[str]]:
    docs: list[dict[str, str]] = []
    errors: list[str] = []

    for index, raw_path in enumerate(patent_paths[:max_patent_files], start=1):
        path = Path(raw_path).expanduser()
        if not path.exists() or not path.is_file():
            errors.append(f"file not found: {raw_path}")
            continue

        try:
            text = read_patent_text(path)
        except Exception as exc:
            errors.append(f"failed reading file: {raw_path} ({exc})")
            continue

        if not text.strip():
            errors.append(f"empty or unreadable content: {raw_path}")
            continue

        docs.append(
            {
                "id": f"patent-{index}",
                "path": str(path.resolve()),
                "title": guess_title(path, text),
                "text": trim_text(text, max_patent_text_chars),
            }
        )
    return docs, errors


def read_patent_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf_text(path)
    if suffix == ".docx":
        return read_docx_text(path)
    if suffix == ".doc":
        return read_doc_text(path)

    text = path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".json":
        try:
            parsed = json.loads(text)
            return json_to_text(parsed)
        except json.JSONDecodeError:
            return text
    if suffix in {".xml", ".html", ".htm"}:
        cleaned = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"\s+", " ", cleaned).strip()

    return text


def read_docx_text(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            xml_data = archive.read("word/document.xml")
    except Exception:
        return ""

    try:
        root = ElementTree.fromstring(xml_data)
    except Exception:
        return ""

    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs: list[str] = []
    for paragraph in root.iter(f"{ns}p"):
        runs = [node.text for node in paragraph.iter(f"{ns}t") if node.text]
        if runs:
            paragraphs.append("".join(runs))
    return "\n".join(paragraphs)


def read_doc_text(path: Path) -> str:
    # Strategy 1: olefile — pure-Python, cross-platform .doc parser
    text = _read_doc_via_olefile(path)
    if text:
        return text

    # Strategy 2: antiword / catdoc — Unix CLI tools
    for tool in ("antiword", "catdoc"):
        if not shutil.which(tool):
            continue
        try:
            completed = subprocess.run(
                [tool, str(path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception:
            continue
        output = (completed.stdout or "").strip()
        if completed.returncode == 0 and output:
            return output

    # Strategy 3: raw read as fallback
    try:
        content = path.read_text(encoding="utf-8", errors="ignore").strip()
        if content and "\x00" not in content:
            return content
    except Exception:
        return ""
    return ""


def _read_doc_via_olefile(path: Path) -> str:
    """Extract text from .doc (OLE2/CFBF) by parsing the Word Binary piece table.
    Works on all platforms via olefile (pure Python)."""
    try:
        import olefile  # type: ignore
    except ImportError:
        return ""

    if not olefile.isOleFile(str(path)):
        return ""

    try:
        ole = olefile.OleFileIO(str(path))
    except Exception:
        return ""

    try:
        if not ole.exists("WordDocument"):
            return ""

        word_stream = ole.openstream("WordDocument").read()
        if len(word_stream) < 12:
            return ""

        # FIB: bytes 0x000A-0x000B → flags; bit 9 of wIdent area tells table name
        fib_flags = int.from_bytes(word_stream[0x000A:0x000C], "little")
        table_name = "1Table" if (fib_flags & 0x0200) else "0Table"

        if not ole.exists(table_name):
            return ""

        table_stream = ole.openstream(table_name).read()

        # FIB: ccpText at offset 0x004C (4 bytes LE) — character count of main text
        ccp_text = int.from_bytes(word_stream[0x004C:0x0050], "little")
        if ccp_text <= 0 or ccp_text > 10_000_000:
            return ""

        # FIB: fcClx at 0x01A2 (4 bytes), lcbClx at 0x01A6 (4 bytes)
        fc_clx = int.from_bytes(word_stream[0x01A2:0x01A6], "little")
        lcb_clx = int.from_bytes(word_stream[0x01A6:0x01AA], "little")

        if fc_clx == 0 or lcb_clx == 0:
            return ""

        clx = table_stream[fc_clx : fc_clx + lcb_clx]

        # Parse CLX: skip Prcs (type 0x01), find Pcdt (type 0x02)
        pos = 0
        while pos < len(clx):
            clx_type = clx[pos]
            if clx_type == 0x01:
                # Prc: 1 byte type + 2 bytes size + data
                if pos + 3 > len(clx):
                    break
                prc_size = int.from_bytes(clx[pos + 1 : pos + 3], "little")
                pos += 3 + prc_size
            elif clx_type == 0x02:
                # Pcdt: 1 byte type + 4 bytes size + PlcPcd
                if pos + 5 > len(clx):
                    break
                pcdt_size = int.from_bytes(clx[pos + 1 : pos + 5], "little")
                plc_pcd = clx[pos + 5 : pos + 5 + pcdt_size]
                return _extract_text_from_plcpcd(plc_pcd, ccp_text, word_stream)
            else:
                break

    except Exception:
        pass
    finally:
        ole.close()

    return ""


def _extract_text_from_plcpcd(
    plc_pcd: bytes, ccp_text: int, word_stream: bytes
) -> str:
    """Parse PlcPcd structure to extract text pieces from WordDocument stream."""
    # PlcPcd = array of (n+1) CPs (4 bytes each) + array of n PCDs (8 bytes each)
    # Solve: (n+1)*4 + n*8 = len(plc_pcd) → n = (len(plc_pcd) - 4) / 12
    n_pieces = (len(plc_pcd) - 4) // 12
    if n_pieces <= 0:
        return ""

    cps: list[int] = []
    for i in range(n_pieces + 1):
        cp = int.from_bytes(plc_pcd[i * 4 : i * 4 + 4], "little")
        cps.append(cp)

    pcd_offset = (n_pieces + 1) * 4
    text_parts: list[str] = []

    for i in range(n_pieces):
        cp_start = cps[i]
        cp_end = cps[i + 1]
        if cp_start >= ccp_text:
            break
        cp_end = min(cp_end, ccp_text)
        char_count = cp_end - cp_start

        pcd_base = pcd_offset + i * 8
        if pcd_base + 8 > len(plc_pcd):
            break

        # PCD: 2 bytes abfcNotUsed + 4 bytes fc + 2 bytes prm
        fc_raw = int.from_bytes(plc_pcd[pcd_base + 2 : pcd_base + 6], "little")

        # Bit 30 of fc → if set, text is CP1252 (1 byte/char); else UTF-16-LE (2 bytes/char)
        is_ansi = bool(fc_raw & 0x40000000)
        fc = fc_raw & 0x3FFFFFFF

        if is_ansi:
            fc = fc // 2  # ANSI offset is halved in the fc field
            byte_start = fc
            byte_end = fc + char_count
            if byte_end > len(word_stream):
                byte_end = len(word_stream)
            chunk = word_stream[byte_start:byte_end].decode("cp1252", errors="ignore")
        else:
            byte_start = fc
            byte_end = fc + char_count * 2
            if byte_end > len(word_stream):
                byte_end = len(word_stream)
            chunk = word_stream[byte_start:byte_end].decode("utf-16-le", errors="ignore")

        text_parts.append(chunk)

    raw_text = "".join(text_parts)
    # Word uses special chars: \r=paragraph, \x07=cell/row end, \x0c=page break
    raw_text = raw_text.replace("\r", "\n").replace("\x07", "\t").replace("\x0c", "\n")
    # Clean up control characters but keep CJK and standard chars
    cleaned = re.sub(r"[\x00-\x08\x0b\x0e-\x1f]", "", raw_text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def read_pdf_text(path: Path) -> str:
    try:
        import fitz  # type: ignore

        doc = fitz.open(str(path))
        chunks = [page.get_text() for page in doc]
        doc.close()
        return "\n".join(chunks)
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore

            reader = PdfReader(str(path))
            chunks: list[str] = []
            for page in reader.pages:
                text = page.extract_text() or ""
                if text:
                    chunks.append(text)
            return "\n".join(chunks)
        except Exception:
            return ""


def json_to_text(data: Any, max_items: int = 200) -> str:
    parts: list[str] = []

    def walk(value: Any) -> None:
        if len(parts) >= max_items:
            return
        if isinstance(value, dict):
            for key, val in value.items():
                if len(parts) >= max_items:
                    break
                parts.append(str(key))
                walk(val)
            return
        if isinstance(value, list):
            for item in value:
                if len(parts) >= max_items:
                    break
                walk(item)
            return

        token = str(value).strip()
        if token:
            parts.append(token)

    walk(data)
    return "\n".join(parts)


def guess_title(path: Path, text: str) -> str:
    for line in text.splitlines()[:20]:
        token = line.strip()
        if token and len(token) <= 160:
            return token
    return path.stem


_ABSTRACT_MARKERS = re.compile(
    r"(?:摘\s*要|abstract|summary\s+of\s+the\s+invention|技术领域|technical\s+field)",
    re.IGNORECASE,
)
_CLAIMS_MARKERS = re.compile(
    r"(?:权\s*利\s*要\s*求\s*书?|claims?)\s*[:：]?\s*\n",
    re.IGNORECASE,
)


def _extract_heuristic_summary(text: str, max_chars: int = 1500) -> str:
    """Best-effort summary extraction for non-preprocessed patent text.

    Tries to locate abstract / claims sections via common markers instead
    of blindly taking the first N characters (which are often boilerplate).
    """
    if not text or not text.strip():
        return text[:max_chars] if text else ""

    parts: list[str] = []

    am = _ABSTRACT_MARKERS.search(text)
    if am:
        after = text[am.start():]
        parts.append(after[:800])

    cm = _CLAIMS_MARKERS.search(text)
    if cm:
        after = text[cm.end():]
        parts.append(after[:700])

    if parts:
        combined = "\n\n".join(parts)
        if len(combined) >= 200:
            return combined[:max_chars]

    return text[:max_chars]


def trim_text(text: str, max_chars: int) -> str:
    normalized = text.replace("\r\n", "\n").strip()
    if len(normalized) <= max_chars:
        return normalized
    head = max_chars // 2
    tail = max_chars - head
    return f"{normalized[:head]}\n\n...[truncated]...\n\n{normalized[-tail:]}"


def to_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        token = value.strip()
        return [token] if token else []
    if isinstance(value, list):
        output: list[str] = []
        for item in value:
            token = str(item).strip()
            if token:
                output.append(token)
        return output
    token = str(value).strip()
    return [token] if token else []


def to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _try_repair_truncated_json(fragment: str) -> dict[str, Any] | None:
    """Best-effort repair of truncated JSON by closing open brackets/braces."""
    opens = 0
    open_sq = 0
    in_string = False
    escaped = False
    for ch in fragment:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            opens += 1
        elif ch == "}":
            opens -= 1
        elif ch == "[":
            open_sq += 1
        elif ch == "]":
            open_sq -= 1

    if opens <= 0 and open_sq <= 0:
        return None

    if in_string:
        fragment += '"'
    fragment += "]" * max(open_sq, 0)
    fragment += "}" * max(opens, 0)
    try:
        parsed = json.loads(fragment)
        if isinstance(parsed, dict):
            print("[info] repaired truncated JSON successfully")
            return parsed
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None


def extract_json_payload(text: str | None) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    for block in re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw, flags=re.IGNORECASE):
        try:
            parsed = json.loads(block)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    decoder = json.JSONDecoder()
    for index, ch in enumerate(raw):
        if ch != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    first_brace = raw.find("{")
    if first_brace >= 0:
        fragment = raw[first_brace:]
        repaired = _try_repair_truncated_json(fragment)
        if repaired is not None:
            return repaired

    return None


def normalize_patterns(raw_patterns: Any, *, max_patterns: int = MAX_PATTERNS) -> list[dict[str, Any]]:
    if not isinstance(raw_patterns, list):
        return []

    all_parsed: list[dict[str, Any]] = []
    for index, item in enumerate(raw_patterns, start=1):
        if not isinstance(item, dict):
            continue

        score = item.get("score") if isinstance(item.get("score"), dict) else {}
        distinctiveness = to_int(score.get("distinctiveness"), 0)
        sophistication = to_int(score.get("sophistication"), 0)
        system_impact = to_int(score.get("system_impact"), 0)
        frame_shift = to_int(score.get("frame_shift"), 0)
        total = to_int(score.get("total"), distinctiveness + sophistication + system_impact + frame_shift)

        pattern = {
            "pattern_id": str(item.get("pattern_id") or f"pattern-{index}").strip(),
            "title": str(item.get("title") or f"Pattern {index}").strip(),
            "category": str(item.get("category") or "other").strip().lower(),
            "description": str(item.get("description") or "").strip(),
            "source_files": to_str_list(item.get("source_files")),
            "score": {
                "distinctiveness": distinctiveness,
                "sophistication": sophistication,
                "system_impact": system_impact,
                "frame_shift": frame_shift,
                "total": total,
            },
            "score_total": total,
            "why_distinctive": str(item.get("why_distinctive") or "").strip(),
            "patent_signals": item.get("patent_signals") if isinstance(item.get("patent_signals"), dict) else {},
            "claim_angles": to_str_list(item.get("claim_angles")),
            "abstract_mechanism": str(item.get("abstract_mechanism") or "").strip(),
            "concrete_reference": str(item.get("concrete_reference") or "").strip(),
        }
        all_parsed.append(pattern)

    all_parsed.sort(key=lambda x: x.get("score_total", 0), reverse=True)
    return all_parsed[:max_patterns]


def normalize_matches(raw_matches: Any, *, max_matches: int = MAX_MATCHES) -> list[dict[str, Any]]:
    if not isinstance(raw_matches, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in raw_matches:
        if not isinstance(item, dict):
            continue

        similarity = to_float(item.get("similarity"), 0.0)
        similarity = max(0.0, min(1.0, similarity))

        match_level = str(item.get("match_level") or "").strip().lower()
        if match_level not in {"high", "medium", "low", "none"}:
            if similarity >= 0.75:
                match_level = "high"
            elif similarity >= 0.6:
                match_level = "medium"
            elif similarity >= 0.4:
                match_level = "low"
            else:
                match_level = "none"

        confidence = str(item.get("confidence") or "medium").strip().lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "medium"

        normalized.append(
            {
                "pattern_id": str(item.get("pattern_id") or "").strip(),
                "pattern_title": str(item.get("pattern_title") or "").strip(),
                "patent_file": str(item.get("patent_file") or "").strip(),
                "patent_title": str(item.get("patent_title") or "").strip(),
                "match_level": match_level,
                "similarity": similarity,
                "confidence": confidence,
                "reason": str(item.get("reason") or "").strip(),
                "evidence": to_str_list(item.get("evidence")),
            }
        )

    normalized.sort(key=lambda x: x["similarity"], reverse=True)
    return normalized[:max_matches]


def is_actionable_match(match: dict[str, Any]) -> bool:
    level = str(match.get("match_level") or "")
    similarity = to_float(match.get("similarity"), 0.0)
    if level == "high":
        return True
    if level == "medium" and similarity >= 0.6:
        return True
    if similarity >= 0.8:
        return True
    return False


def match_level_rank(value: str) -> int:
    mapping = {"none": 0, "low": 1, "medium": 2, "high": 3}
    return mapping.get(str(value or "").strip().lower(), 0)


def confidence_rank(value: str) -> int:
    mapping = {"low": 1, "medium": 2, "high": 3}
    return mapping.get(str(value or "").strip().lower(), 0)


def display_file(value: str) -> str:
    token = str(value or "").strip()
    if not token:
        return "-"
    return patent_file_basename(token) or token


def patent_file_basename(value: str) -> str:
    token = str(value or "").strip()
    if not token:
        return ""
    parts = re.split(r"[\\/]+", token)
    return parts[-1] if parts else token


def normalize_patent_token(value: str) -> str:
    token = str(value or "").strip().lower()
    if not token:
        return ""
    return re.sub(r"\s+", " ", token)


def build_patent_doc_index_maps(
    patent_docs: list[dict[str, str]],
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    by_path: dict[str, int] = {}
    by_file_name: dict[str, int] = {}
    by_title: dict[str, int] = {}

    for index, doc in enumerate(patent_docs):
        path_token = normalize_patent_token(doc.get("path") or "")
        if path_token and path_token not in by_path:
            by_path[path_token] = index

        file_name = normalize_patent_token(patent_file_basename(doc.get("path") or ""))
        if file_name and file_name not in by_file_name:
            by_file_name[file_name] = index

        title_token = normalize_patent_token(doc.get("title") or "")
        if title_token and title_token not in by_title:
            by_title[title_token] = index

    return by_path, by_file_name, by_title


def resolve_patent_doc_index(
    *,
    patent_file: str,
    patent_title: str,
    by_path: dict[str, int],
    by_file_name: dict[str, int],
    by_title: dict[str, int],
) -> int | None:
    file_token = normalize_patent_token(patent_file)
    if file_token and file_token in by_path:
        return by_path[file_token]

    base_token = normalize_patent_token(patent_file_basename(patent_file))
    if base_token and base_token in by_file_name:
        return by_file_name[base_token]

    title_token = normalize_patent_token(patent_title)
    if title_token and title_token in by_title:
        return by_title[title_token]

    if file_token and file_token in by_title:
        return by_title[file_token]
    return None


def group_matches_by_patent(
    matches: list[dict[str, Any]],
    pattern_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}

    for match in matches:
        patent_file = str(match.get("patent_file") or "").strip()
        patent_title = str(match.get("patent_title") or "").strip() or display_file(patent_file)
        patent_key = patent_file or patent_title or "unknown"

        group = groups.get(patent_key)
        if group is None:
            group = {
                "patent_key": patent_key,
                "patent_file": patent_file,
                "patent_title": patent_title or "unknown",
                "match_count": 0,
                "actionable_count": 0,
                "weak_count": 0,
                "max_similarity": 0.0,
                "max_match_level": "none",
                "max_confidence": "low",
                "matches": [],
            }
            groups[patent_key] = group

        pattern_id = str(match.get("pattern_id") or "").strip()
        pattern = pattern_map.get(pattern_id, {})
        enriched_match = {
            **match,
            "pattern_source_files": to_str_list(pattern.get("source_files")),
            "pattern_description": str(pattern.get("description") or "").strip(),
            "pattern_abstract_mechanism": str(pattern.get("abstract_mechanism") or "").strip(),
        }

        group["matches"].append(enriched_match)
        group["match_count"] += 1
        group["max_similarity"] = max(group["max_similarity"], to_float(match.get("similarity"), 0.0))

        match_level = str(match.get("match_level") or "none").strip().lower()
        if match_level_rank(match_level) > match_level_rank(group["max_match_level"]):
            group["max_match_level"] = match_level

        confidence = str(match.get("confidence") or "low").strip().lower()
        if confidence_rank(confidence) > confidence_rank(group["max_confidence"]):
            group["max_confidence"] = confidence

        if is_actionable_match(match):
            group["actionable_count"] += 1
        elif match_level != "none":
            group["weak_count"] += 1

    grouped = list(groups.values())
    for item in grouped:
        item["matches"].sort(
            key=lambda x: (
                to_float(x.get("similarity"), 0.0),
                match_level_rank(str(x.get("match_level") or "")),
                confidence_rank(str(x.get("confidence") or "")),
            ),
            reverse=True,
        )

    grouped.sort(
        key=lambda x: (
            x.get("actionable_count", 0),
            to_float(x.get("max_similarity"), 0.0),
            x.get("match_count", 0),
        ),
        reverse=True,
    )
    return grouped


def build_preprocessor_prompt(
    skill: SkillLoadResult,
    patents_batch: list[dict[str, str]],
) -> str:
    """Build prompt for AI patent structure extraction."""
    sections = []
    for item in patents_batch:
        pid = item.get("id", "unknown")
        text = item.get("text", "")
        sections.append(f"--- PATENT [{pid}] ---\n{text}")
    joined = "\n\n".join(sections)
    return (
        "Follow the patent-preprocessor skill exactly.\n"
        "Extract structured metadata from each patent below.\n"
        "Output must be a JSON object only. No markdown, no explanation.\n\n"
        f"[patent-preprocessor skill]\n{skill.content}\n\n"
        f"[patent texts]\n{joined}\n\n"
        "Return this JSON shape:\n"
        "{\n"
        '  "patents": [\n'
        "    {\n"
        '      "id": "patent-1",\n'
        '      "title": "...",\n'
        '      "abstract": "...",\n'
        '      "independent_claims": ["..."],\n'
        '      "keywords": ["..."]\n'
        "    }\n"
        "  ]\n"
        "}"
    )


async def preprocess_patents(
    patent_paths: list[str],
    cloud_client: "SimpleClaudeCodeClient",
    preprocessor_skill: SkillLoadResult,
    cache: PatentCache,
    *,
    max_patent_text_chars: int = MAX_PATENT_TEXT_CHARS,
    batch_size: int = PREPROCESS_BATCH_SIZE,
    on_progress: Callable | None = None,
) -> tuple[list[ParsedPatent], list[str]]:
    """Preprocess patents with AI extraction + cache. Returns (parsed_list, errors)."""
    results: list[ParsedPatent] = []
    errors: list[str] = []
    need_ai: list[dict[str, Any]] = []

    for idx, raw_path in enumerate(patent_paths, start=1):
        path = Path(raw_path).expanduser()
        if not path.exists() or not path.is_file():
            errors.append(f"file not found: {raw_path}")
            continue

        try:
            file_hash = compute_file_hash(path)
        except Exception as exc:
            errors.append(f"failed hashing: {raw_path} ({exc})")
            continue

        cached = cache.get(file_hash)
        if cached is not None:
            cached.path = str(path.resolve())
            results.append(cached)
            if on_progress:
                _r = on_progress(idx, len(patent_paths), f"cache hit: {path.name}")
                if asyncio.isfuture(_r) or asyncio.iscoroutine(_r):
                    await _r
            continue

        try:
            text = read_patent_text(path)
        except Exception as exc:
            errors.append(f"failed reading: {raw_path} ({exc})")
            continue

        if not text.strip():
            errors.append(f"empty content: {raw_path}")
            continue

        need_ai.append({
            "index": idx,
            "path": str(path.resolve()),
            "file_hash": file_hash,
            "raw_title": guess_title(path, text),
            "full_text": text,
            "text_for_ai": trim_text(text, max_patent_text_chars),
        })

    total = len(patent_paths)
    processed_count = len(results)

    for batch_start in range(0, len(need_ai), batch_size):
        batch = need_ai[batch_start : batch_start + batch_size]
        batch_input = [
            {"id": f"patent-{item['index']}", "text": item["text_for_ai"]}
            for item in batch
        ]

        prompt = build_preprocessor_prompt(preprocessor_skill, batch_input)
        try:
            workspace = Path(batch[0]["path"]).parent
            raw_response = await cloud_client.analyze(
                message=prompt,
                system_prompt="You are a patent document parser. Reply with valid JSON object only.",
                workspace_path=workspace,
                timeout=300,
                idle_timeout=IDLE_TIMEOUT,
            )
            payload = extract_json_payload(raw_response)
            ai_patents = (payload or {}).get("patents", []) if payload else []
        except Exception as exc:
            ai_patents = []
            for item in batch:
                errors.append(f"AI preprocessing failed for {Path(item['path']).name}: {exc}")

        ai_map: dict[str, dict[str, Any]] = {}
        if isinstance(ai_patents, list):
            for ap in ai_patents:
                if isinstance(ap, dict):
                    ap_id = str(ap.get("id") or "").strip()
                    if ap_id:
                        ai_map[ap_id] = ap

        for item in batch:
            pid = f"patent-{item['index']}"
            ai_data = ai_map.get(pid, {})

            parsed = ParsedPatent(
                file_hash=item["file_hash"],
                path=item["path"],
                raw_title=item["raw_title"],
                title=str(ai_data.get("title") or item["raw_title"]).strip(),
                abstract=str(ai_data.get("abstract") or "").strip(),
                independent_claims=to_str_list(ai_data.get("independent_claims")),
                keywords=to_str_list(ai_data.get("keywords")),
                full_text=item["full_text"],
            )
            parsed.summary_text = parsed.build_summary_text()
            cache.put(parsed)
            results.append(parsed)
            processed_count += 1
            if on_progress:
                _r = on_progress(processed_count, total, f"processed: {Path(item['path']).name}")
                if asyncio.isfuture(_r) or asyncio.iscoroutine(_r):
                    await _r

    return results, errors


# ---------------------------------------------------------------------------
# Layer 1: BM25 fast recall
# ---------------------------------------------------------------------------

_JIEBA_AVAILABLE: bool | None = None


def _simple_tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese/English text for BM25.

    Uses jieba search-mode segmentation when available; falls back to
    character bigrams for Chinese text so that multi-char terms still
    carry discriminative power (single-char tokens are nearly useless
    for BM25 because their IDF is extremely low).
    """
    global _JIEBA_AVAILABLE
    if _JIEBA_AVAILABLE is None:
        try:
            import jieba  # type: ignore
            jieba.setLogLevel(jieba.logging.WARNING)
            _JIEBA_AVAILABLE = True
        except ImportError:
            _JIEBA_AVAILABLE = False

    if _JIEBA_AVAILABLE:
        return _tokenize_jieba(text)
    return _tokenize_bigram_fallback(text)


_STOPWORDS = frozenset({
    "the", "and", "for", "that", "this", "with", "from", "are", "was",
    "were", "been", "being", "have", "has", "had", "not", "but", "its",
    "also", "such", "can", "may", "will", "each", "one", "two",
    "method", "system", "device", "apparatus", "comprising", "wherein",
    "thereof", "herein", "according", "based", "includes", "including",
    "provided", "present", "described", "related", "using", "used",
    "一种", "所述", "其中", "包括", "根据", "具有", "用于", "通过",
    "以及", "或者", "并且", "其特征在于", "本发明", "技术",
})


def _tokenize_jieba(text: str) -> list[str]:
    import jieba  # type: ignore

    tokens: list[str] = []
    for word in jieba.cut_for_search(text):
        w = word.strip().lower()
        if len(w) >= 2 and not w.isspace() and w not in _STOPWORDS:
            tokens.append(w)
    return tokens


def _tokenize_bigram_fallback(text: str) -> list[str]:
    """Fallback tokenizer: English words + Chinese bigrams."""
    tokens: list[str] = []
    text_lower = text.lower()
    ascii_buf: list[str] = []
    cn_buf: list[str] = []

    def _flush_ascii() -> None:
        if ascii_buf:
            word = "".join(ascii_buf)
            ascii_buf.clear()
            if len(word) >= 2 and word not in _STOPWORDS:
                tokens.append(word)

    def _flush_cn() -> None:
        if cn_buf:
            for i in range(len(cn_buf)):
                if i + 1 < len(cn_buf):
                    tokens.append(cn_buf[i] + cn_buf[i + 1])
                if i + 2 < len(cn_buf):
                    tokens.append(cn_buf[i] + cn_buf[i + 1] + cn_buf[i + 2])
            cn_buf.clear()

    for ch in text_lower:
        if ch.isascii():
            _flush_cn()
            if ch.isalnum():
                ascii_buf.append(ch)
            else:
                _flush_ascii()
        else:
            _flush_ascii()
            cn_buf.append(ch)

    _flush_ascii()
    _flush_cn()
    return tokens


def bm25_recall(
    patterns: list[dict[str, Any]],
    patent_docs: list[ParsedPatent],
    top_k: int = DEFAULT_RECALL_TOP_K,
) -> list[ParsedPatent]:
    """BM25 recall: per-pattern queries with max-score aggregation.

    Instead of merging all patterns into one giant query (which dilutes
    signal), we query once per pattern and keep the *maximum* score each
    patent received across all patterns.  A patent only needs to be
    relevant to ONE pattern to be recalled.

    Patent keywords (from preprocessing) are appended to the corpus text
    with repetition to boost their BM25 weight.
    """
    if len(patent_docs) <= top_k:
        return list(patent_docs)

    try:
        from rank_bm25 import BM25Okapi  # type: ignore
    except ImportError:
        print("[warn] rank_bm25 not installed, skipping BM25 recall")
        return list(patent_docs)

    corpus_texts: list[str] = []
    for p in patent_docs:
        base = p.summary_text or p.full_text[:2000]
        kw_boost = (" ".join(p.keywords) + " ") * 3 if p.keywords else ""
        corpus_texts.append(f"{base} {kw_boost}")
    tokenized_corpus = [_simple_tokenize(t) for t in corpus_texts]

    if not any(tokenized_corpus):
        return list(patent_docs)

    bm25 = BM25Okapi(tokenized_corpus)

    n = len(patent_docs)
    aggregated = [0.0] * n

    for pat in patterns:
        query_text = " ".join(filter(None, [
            str(pat.get("title") or ""),
            str(pat.get("abstract_mechanism") or ""),
            str(pat.get("why_distinctive") or ""),
            str(pat.get("description") or ""),
            *to_str_list(pat.get("claim_angles")),
        ]))
        query_tokens = _simple_tokenize(query_text)
        if not query_tokens:
            continue
        scores = bm25.get_scores(query_tokens)
        for i in range(n):
            if scores[i] > aggregated[i]:
                aggregated[i] = scores[i]

    indexed_scores = sorted(enumerate(aggregated), key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in indexed_scores[:top_k]]
    return [patent_docs[i] for i in top_indices]


# ---------------------------------------------------------------------------
# Layer 2: Light LLM reranking
# ---------------------------------------------------------------------------

def build_light_validator_prompt(
    skill: SkillLoadResult,
    patterns: list[dict[str, Any]],
    patent_summaries: list[dict[str, str]],
) -> str:
    """Build a compact prompt using only patent summaries for fast LLM screening."""
    patterns_compact = []
    for p in patterns[:MAX_PATTERNS]:
        patterns_compact.append({
            "pattern_id": p.get("pattern_id", ""),
            "title": p.get("title", ""),
            "abstract_mechanism": p.get("abstract_mechanism", ""),
            "claim_angles": p.get("claim_angles", []),
        })
    patterns_json = json.dumps(patterns_compact, ensure_ascii=False)

    summary_sections = []
    for ps in patent_summaries:
        summary_sections.append(
            f"### [{ps['id']}] {ps['title']}\n{ps['summary']}"
        )
    summaries_joined = "\n\n".join(summary_sections)

    return (
        "You are screening patents for TECHNICAL METHOD relevance to code implementation patterns.\n\n"
        "MATCHING GUIDANCE:\n"
        "- Focus on the underlying technical APPROACH, ALGORITHM, or ARCHITECTURE, not surface keywords.\n"
        "- A patent about 'gradient-based optimization' IS relevant to a pattern about 'loss minimization via backpropagation'.\n"
        "- A patent about 'data caching' is NOT relevant to a pattern about 'data encryption' even though both mention 'data'.\n"
        "- Consider functional equivalence: different terminology describing the same technical method counts as a match.\n\n"
        "SIMILARITY SCALE:\n"
        "- 0.0-0.2: No technical overlap in method or approach\n"
        "- 0.2-0.4: Same general domain but fundamentally different approach\n"
        "- 0.4-0.6: Similar technical approach with notable differences\n"
        "- 0.6-0.8: Closely related technical method or architecture\n"
        "- 0.8-1.0: Nearly identical technical approach\n\n"
        "Include ALL patents with estimated_similarity >= 0.15.\n"
        "Output must be a JSON object only. No markdown, no explanation.\n\n"
        f"[code-patent-validator skill]\n{skill.content}\n\n"
        "[patterns (compact)]\n"
        f"{patterns_json}\n\n"
        "[patent summaries]\n"
        f"{summaries_joined}\n\n"
        "Return this JSON:\n"
        "{\n"
        '  "rankings": [\n'
        "    {\n"
        '      "patent_id": "patent-xxx",\n'
        '      "patent_title": "...",\n'
        '      "estimated_similarity": 0.0,\n'
        '      "matched_pattern_ids": ["pattern-1"],\n'
        '      "brief_reason": "..."\n'
        "    }\n"
        "  ]\n"
        "}"
    )


async def light_rerank(
    cloud_client: "SimpleClaudeCodeClient",
    validator_skill: SkillLoadResult,
    patterns: list[dict[str, Any]],
    candidates: list[ParsedPatent],
    top_n: int,
    workspace_path: Path,
    timeout: int,
    idle_timeout: int = IDLE_TIMEOUT,
) -> list[ParsedPatent]:
    """Layer 2: Light LLM reranking using summaries only."""
    if len(candidates) <= top_n:
        return list(candidates)

    patent_summaries = []
    id_map: dict[str, ParsedPatent] = {}
    for i, p in enumerate(candidates):
        pid = f"patent-{p.file_hash[:8]}"
        patent_summaries.append({
            "id": pid,
            "title": p.title or p.raw_title,
            "summary": (p.summary_text or p.full_text[:1500])[:1500],
        })
        id_map[pid] = p

    prompt = build_light_validator_prompt(validator_skill, patterns, patent_summaries)

    try:
        raw = await cloud_client.analyze(
            message=prompt,
            system_prompt="You are a patent relevance screener. Reply with valid JSON object only.",
            workspace_path=workspace_path,
            timeout=timeout,
            idle_timeout=idle_timeout,
        )
        payload = extract_json_payload(raw)
    except Exception as exc:
        print(f"[warn] light rerank failed: {exc}, keeping all candidates")
        return candidates[:top_n]

    if not payload:
        return candidates[:top_n]

    rankings = payload.get("rankings", [])
    if not isinstance(rankings, list):
        return candidates[:top_n]

    ranked: list[tuple[float, ParsedPatent]] = []
    seen_ids: set[str] = set()
    for item in rankings:
        if not isinstance(item, dict):
            continue
        pid = str(item.get("patent_id") or "").strip()
        sim = to_float(item.get("estimated_similarity"), 0.0)
        if pid in id_map and pid not in seen_ids:
            seen_ids.add(pid)
            ranked.append((sim, id_map[pid]))

    ranked.sort(key=lambda x: x[0], reverse=True)
    result = [p for _, p in ranked[:top_n]]

    for p in candidates:
        pid = f"patent-{p.file_hash[:8]}"
        if pid not in seen_ids and len(result) < top_n:
            result.append(p)

    return result[:top_n]


# ---------------------------------------------------------------------------
# Layer 3: Deep concurrent validation
# ---------------------------------------------------------------------------

async def run_deep_validation(
    cloud_client: "SimpleClaudeCodeClient",
    validator_skill: SkillLoadResult,
    patterns: list[dict[str, Any]],
    patents: list[ParsedPatent],
    workspace_path: Path,
    timeout: int,
    max_concurrency: int = DEFAULT_DEEP_MAX_CONCURRENCY,
    idle_timeout: int = IDLE_TIMEOUT,
) -> list[dict[str, Any]]:
    """Layer 3: Deep validation - one patent per LLM call, concurrent."""
    if not patents or not patterns:
        return []

    semaphore = asyncio.Semaphore(max_concurrency)
    all_matches: list[dict[str, Any]] = []
    lock = asyncio.Lock()

    async def validate_one(patent: ParsedPatent, idx: int) -> None:
        async with semaphore:
            doc = patent.to_legacy_doc()
            prompt = build_validator_prompt(validator_skill, patterns, [doc])
            try:
                raw = await cloud_client.analyze(
                    message=prompt,
                    system_prompt="You are a strict patent pattern validator. Reply with valid JSON object only.",
                    workspace_path=workspace_path,
                    timeout=timeout,
                    idle_timeout=idle_timeout,
                )
                payload = extract_json_payload(raw)
                matches = (payload or {}).get("matches", []) if payload else []
                if isinstance(matches, list):
                    async with lock:
                        all_matches.extend(matches)
                print(f"[info] deep validation {idx+1}/{len(patents)}: {patent.title or patent.raw_title} -> {len(matches)} matches")
            except Exception as exc:
                print(f"[warn] deep validation failed for {patent.title or patent.raw_title}: {exc}")

    tasks = [validate_one(p, i) for i, p in enumerate(patents)]
    await asyncio.gather(*tasks)
    return all_matches


def build_scanner_prompt(skill: SkillLoadResult, repo_path: Path) -> str:
    return (
        "Follow the provided code-patent-scanner skill exactly.\n"
        "You MUST inspect the repository by yourself using available tools in this workspace.\n"
        "Do not rely on pre-selected snippets; discover and prioritize important files autonomously.\n"
        "Output must be a JSON object only. No markdown, no explanation.\n\n"
        f"[code-patent-scanner skill]\n{skill.content}\n\n"
        "[repository context]\n"
        f"repository: {repo_path}\n"
        "Please explore source files in this repository and then extract technical patterns.\n\n"
        "Return this JSON shape (you may add fields but keep all core fields):\n"
        "{\n"
        '  "scan_metadata": {"repository": "...", "files_analyzed": 0, "files_skipped": 0},\n'
        '  "patterns": [\n'
        "    {\n"
        '      "pattern_id": "pattern-1",\n'
        '      "title": "...",\n'
        '      "category": "algorithmic|architectural|data-structure|integration|other",\n'
        '      "description": "...",\n'
        '      "source_files": ["a/b.py:10-80"],\n'
        '      "score": {"distinctiveness": 0, "sophistication": 0, "system_impact": 0, "frame_shift": 0, "total": 0},\n'
        '      "why_distinctive": "...",\n'
        '      "patent_signals": {"market_demand": "low|medium|high", "competitive_value": "low|medium|high", "novelty_confidence": "low|medium|high"},\n'
        '      "claim_angles": ["..."],\n'
        '      "abstract_mechanism": "...",\n'
        '      "concrete_reference": "..."\n'
        "    }\n"
        "  ],\n"
        '  "summary": {"total_patterns": 0}\n'
        "}"
    )


def build_validator_prompt(
    skill: SkillLoadResult,
    patterns: list[dict[str, Any]],
    patent_docs: list[dict[str, str]],
) -> str:
    patterns_json = json.dumps(patterns[:MAX_PATTERNS], ensure_ascii=False)
    patent_sections = []
    for patent in patent_docs:
        patent_sections.append(f"### {patent['title']} ({patent['path']})\n{patent['text']}")

    patent_sections_joined = "\n\n".join(patent_sections)
    return (
        "Follow the provided code-patent-validator skill exactly and match patterns to patent docs.\n"
        "Output must be a JSON object only. No markdown, no explanation.\n\n"
        f"[code-patent-validator skill]\n{skill.content}\n\n"
        "[pattern input]\n"
        f"patterns: {patterns_json}\n\n"
        "[uploaded patent text]\n"
        f"{patent_sections_joined}\n\n"
        "Return this JSON shape (you may add fields but keep all core fields):\n"
        "{\n"
        '  "matches": [\n'
        "    {\n"
        '      "pattern_id": "pattern-1",\n'
        '      "pattern_title": "...",\n'
        '      "patent_file": "...",\n'
        '      "patent_title": "...",\n'
        '      "match_level": "high|medium|low|none",\n'
        '      "similarity": 0.0,\n'
        '      "confidence": "high|medium|low",\n'
        '      "reason": "...",\n'
        '      "evidence": ["..."]\n'
        "    }\n"
        "  ],\n"
        '  "summary": "..."\n'
        "}"
    )


async def run_patent_check(options: PatentCheckOptions) -> int:
    async def _emit_stage(stage: str, detail: str = "") -> None:
        if options.on_stage:
            _r = options.on_stage(stage, detail)
            if asyncio.isfuture(_r) or asyncio.iscoroutine(_r):
                await _r

    repo_path = Path(options.repo).expanduser().resolve()
    if not repo_path.exists() or not repo_path.is_dir():
        print(f"[error] repo does not exist or is not a directory: {repo_path}", file=sys.stderr)
        return 2

    patent_paths = normalize_patent_paths(options.patent or [])
    if not patent_paths:
        print("[error] no patent files provided; use --patent PATH", file=sys.stderr)
        return 2

    cloud_client = SimpleClaudeCodeClient(
        model=options.model,
        anthropic_api_key=options.api_key,
        anthropic_base_url=options.base_url,
    )
    available, detail = cloud_client.check_available()
    if not available:
        print(f"[error] cloudcode unavailable: {detail}", file=sys.stderr)
        return 3

    print(f"[info] cloudcode: {detail}")
    print(f"[info] repository: {repo_path}")

    skill_adapter = PatentSkillAdapter(options.skills_root)
    scanner_skill = skill_adapter.load_code_patent_scanner()
    validator_skill = skill_adapter.load_code_patent_validator()
    print(
        "[info] skill(scanner): "
        f"{'local' if scanner_skill.loaded else 'fallback'} | {scanner_skill.source}"
    )
    print(
        "[info] skill(validator): "
        f"{'local' if validator_skill.loaded else 'fallback'} | {validator_skill.source}"
    )
    if scanner_skill.note:
        print(f"[info] scanner note: {scanner_skill.note}")
    if validator_skill.note:
        print(f"[info] validator note: {validator_skill.note}")

    # ── Resolve parsed patents (pre-processed or legacy load) ─────────
    funnel: dict[str, int] = {}
    parsed_patents: list[ParsedPatent] | None = None
    patent_errors: list[str] = []

    if options.parsed_patents and isinstance(options.parsed_patents, list):
        parsed_patents = [p for p in options.parsed_patents if isinstance(p, ParsedPatent)]
        print(f"[info] using pre-processed patents: {len(parsed_patents)}")

    if parsed_patents:
        all_parsed = parsed_patents
        patent_docs = [p.to_legacy_doc() for p in all_parsed]
        funnel["total_patents"] = len(all_parsed)
        funnel["layer0_preprocessed"] = len(all_parsed)
        funnel["layer0_cache_hits"] = len(all_parsed)
    else:
        legacy_docs, patent_errors = load_patent_documents(
            patent_paths,
            max_patent_files=options.max_patent_files,
            max_patent_text_chars=options.max_patent_text_chars,
        )
        patent_docs = legacy_docs
        all_parsed = [
            ParsedPatent(
                file_hash=hashlib.sha256(d.get("text", "").encode()).hexdigest()[:16],
                path=d.get("path", ""),
                raw_title=d.get("title", ""),
                title=d.get("title", ""),
                abstract="",
                full_text=d.get("text", ""),
                summary_text=_extract_heuristic_summary(d.get("text", "")),
            )
            for d in legacy_docs
        ]
        funnel["total_patents"] = len(all_parsed)
        funnel["layer0_preprocessed"] = 0
        funnel["layer0_cache_hits"] = 0

    print(f"[info] patent docs loaded: {len(patent_docs)}")
    if patent_errors:
        print(f"[warn] patent doc errors: {len(patent_errors)}")
        for err in patent_errors[:8]:
            print(f"  - {err}")

    if not patent_docs:
        print("[error] all patent files failed to read", file=sys.stderr)
        return 4

    # ── Scanner phase (unchanged) ────────────────────────────────────
    await _emit_stage("scanner", "正在扫描代码仓库，提取技术模式...")
    print("[info] scanner mode: Claude Code autonomous repository reading")
    scanner_prompt = build_scanner_prompt(scanner_skill, repo_path)
    print("[info] running code-patent-scanner ...")
    scanner_raw = await cloud_client.analyze(
        message=scanner_prompt,
        system_prompt="You are a strict code patent pattern analyzer. Reply with valid JSON object only.",
        workspace_path=repo_path,
        timeout=options.timeout,
        idle_timeout=options.idle_timeout,
    )
    scanner_payload = extract_json_payload(scanner_raw)
    if scanner_payload is None:
        preview = (scanner_raw or "")[:500].replace("\n", " ")
        print(f"[warn] scanner output is not valid JSON object", file=sys.stderr)
        print(f"[warn] raw output preview ({len(scanner_raw or '')} chars): {preview}", file=sys.stderr)
        print("[info] retrying scanner with explicit JSON-only instruction ...", flush=True)
        await _emit_stage("scanner_retry", "Scanner 输出非 JSON，重试中...")
        retry_prompt = (
            "Your previous analysis is complete but the output was not valid JSON. "
            "Based on ALL the files you already analyzed, output the final result as a "
            "single raw JSON object NOW. No markdown fences, no explanation, just the JSON.\n\n"
            "Required JSON shape:\n"
            '{"scan_metadata": {"repository": "...", "files_analyzed": 0, "files_skipped": 0}, '
            '"patterns": [{"pattern_id": "pattern-1", "title": "...", "category": "...", '
            '"description": "...", "source_files": ["..."], '
            '"score": {"distinctiveness": 0, "sophistication": 0, "system_impact": 0, '
            '"frame_shift": 0, "total": 0}, '
            '"why_distinctive": "...", "claim_angles": ["..."], '
            '"abstract_mechanism": "...", "concrete_reference": "..."}], '
            '"summary": {"total_patterns": 0}}'
        )
        retry_raw = await cloud_client.analyze(
            message=retry_prompt,
            system_prompt="Output ONLY a valid JSON object. No tools, no markdown, no text.",
            workspace_path=repo_path,
            timeout=min(options.timeout, 120),
            idle_timeout=60,
        )
        scanner_payload = extract_json_payload(retry_raw)
        if scanner_payload is None:
            retry_preview = (retry_raw or "")[:500].replace("\n", " ")
            print(f"[error] retry also not valid JSON", file=sys.stderr)
            print(f"[error] retry preview ({len(retry_raw or '')} chars): {retry_preview}", file=sys.stderr)
            return 6
        print("[info] retry succeeded, got valid JSON")

    patterns = normalize_patterns(scanner_payload.get("patterns"), max_patterns=options.max_patterns)
    print(f"[info] patterns extracted: {len(patterns)} (top by score)")
    for pattern in patterns[:8]:
        print(f"  - {pattern['pattern_id']}: {pattern['title']} (score={pattern['score_total']}/13)")

    if not patterns:
        result = {
            "result": "PASS",
            "detail": "scanner found no technical patterns at all",
            "summary": {
                "uploaded_patent_files": len(patent_docs),
                "files_analyzed": to_int(
                    (scanner_payload.get("scan_metadata") or {}).get("files_analyzed"),
                    0,
                ),
                "patterns": 0,
                "matches": 0,
                "actionable_matches": 0,
                "weak_matches": 0,
                "matched_patent_files": 0,
                "unmatched_patent_files": len(patent_docs),
                "hit_similarity_threshold": PATENT_HIT_SIMILARITY_THRESHOLD,
            },
            "funnel": funnel,
            "patterns": [],
            "matches": [],
            "patent_groups": [],
            "patent_errors": patent_errors,
        }
        write_output_if_requested(result, options.output_json)
        print("[result] PASS: scanner returned zero patterns")
        return 0

    # ── 4-Layer Funnel Validation ─────────────────────────────────────
    use_funnel = len(all_parsed) > 8
    if use_funnel:
        print(f"[info] funnel mode: {len(all_parsed)} patents -> Layer1 BM25 -> Layer2 LLM -> Layer3 deep")

        # Layer 1: BM25 recall
        await _emit_stage("layer1_bm25", f"BM25 快速召回 ({len(all_parsed)} 个专利)")
        print(f"[info] Layer 1: BM25 recall (top_k={options.recall_top_k}) ...")
        recalled = bm25_recall(patterns, all_parsed, top_k=options.recall_top_k)
        funnel["layer1_recalled"] = len(recalled)
        print(f"[info] Layer 1 result: {len(recalled)} patents recalled")

        # Layer 2: Light LLM rerank
        await _emit_stage("layer2_rerank", f"轻量 LLM 复筛 ({len(recalled)} → Top-{options.rerank_top_n})")
        print(f"[info] Layer 2: light LLM rerank (top_n={options.rerank_top_n}) ...")
        reranked = await light_rerank(
            cloud_client, validator_skill, patterns, recalled,
            top_n=options.rerank_top_n,
            workspace_path=repo_path,
            timeout=options.timeout,
            idle_timeout=options.idle_timeout,
        )
        funnel["layer2_reranked"] = len(reranked)
        print(f"[info] Layer 2 result: {len(reranked)} patents after rerank")
        for p in reranked[:8]:
            print(f"  - {p.title or p.raw_title}")

        # Layer 3: Deep concurrent validation
        await _emit_stage("layer3_deep", f"深度比对 ({len(reranked)} 个专利)")
        print(f"[info] Layer 3: deep validation (concurrency={options.deep_max_concurrency}) ...")
        raw_matches = await run_deep_validation(
            cloud_client, validator_skill, patterns, reranked,
            workspace_path=repo_path,
            timeout=options.timeout,
            max_concurrency=options.deep_max_concurrency,
            idle_timeout=options.idle_timeout,
        )
        funnel["layer3_deep_compared"] = len(reranked)
        matches = normalize_matches(raw_matches, max_matches=options.max_matches)

        patent_docs = [p.to_legacy_doc() for p in all_parsed]
    else:
        print(f"[info] direct mode: {len(all_parsed)} patents (<= 8), skipping funnel layers")
        funnel["layer1_recalled"] = len(all_parsed)
        funnel["layer2_reranked"] = len(all_parsed)
        funnel["layer3_deep_compared"] = len(all_parsed)

        await _emit_stage("validator", f"专利比对验证中 ({len(all_parsed)} 个专利)")
        # Layer 3 only: deep validation for all patents
        if len(all_parsed) == 1:
            validator_prompt = build_validator_prompt(validator_skill, patterns, patent_docs)
            print("[info] running code-patent-validator ...")
            validator_raw = await cloud_client.analyze(
                message=validator_prompt,
                system_prompt="You are a strict patent pattern validator. Reply with valid JSON object only.",
                workspace_path=repo_path,
                timeout=options.timeout,
                idle_timeout=options.idle_timeout,
            )
            validator_payload = extract_json_payload(validator_raw)
            if validator_payload is None:
                print("[error] validator output is not valid JSON object", file=sys.stderr)
                return 7
            matches = normalize_matches(validator_payload.get("matches"), max_matches=options.max_matches)
        else:
            raw_matches = await run_deep_validation(
                cloud_client, validator_skill, patterns, all_parsed,
                workspace_path=repo_path,
                timeout=options.timeout,
                max_concurrency=options.deep_max_concurrency,
                idle_timeout=options.idle_timeout,
            )
            matches = normalize_matches(raw_matches, max_matches=options.max_matches)

    # ── Post-processing (unchanged logic) ─────────────────────────────
    await _emit_stage("post_processing", "正在汇总结果...")
    actionable_matches = [item for item in matches if is_actionable_match(item)]
    weak_matches = [item for item in matches if not is_actionable_match(item) and item["match_level"] != "none"]

    pattern_map = {
        str(pattern.get("pattern_id") or "").strip(): pattern
        for pattern in patterns
        if str(pattern.get("pattern_id") or "").strip()
    }
    patent_groups = group_matches_by_patent(matches, pattern_map)
    hit_similarity_threshold = PATENT_HIT_SIMILARITY_THRESHOLD
    by_path, by_file_name, by_title = build_patent_doc_index_maps(patent_docs)
    mapped_hit_patent_indexes: set[int] = set()
    unresolved_hit_keys: set[str] = set()
    for group in patent_groups:
        patent_file = str(group.get("patent_file") or "").strip()
        patent_title = str(group.get("patent_title") or "").strip()
        group["is_hit"] = to_float(group.get("max_similarity"), 0.0) >= hit_similarity_threshold

        matched_index = resolve_patent_doc_index(
            patent_file=patent_file,
            patent_title=patent_title,
            by_path=by_path,
            by_file_name=by_file_name,
            by_title=by_title,
        )
        if matched_index is None:
            if group["is_hit"]:
                unresolved_key = normalize_patent_token(patent_file) or normalize_patent_token(patent_title)
                if unresolved_key:
                    unresolved_hit_keys.add(unresolved_key)
                else:
                    unresolved_hit_keys.add(f"unresolved-hit-{len(unresolved_hit_keys) + 1}")
                group["mapping_status"] = "unresolved"
            continue
        matched_doc = patent_docs[matched_index]
        if not patent_file:
            group["patent_file"] = matched_doc.get("path") or ""
        if not patent_title or patent_title == "unknown":
            group["patent_title"] = matched_doc.get("title") or display_file(group.get("patent_file") or "")
        group["uploaded_patent_path"] = matched_doc.get("path") or ""
        group["uploaded_patent_title"] = matched_doc.get("title") or ""
        group["mapping_status"] = "mapped"

        if group["is_hit"]:
            mapped_hit_patent_indexes.add(matched_index)

    unmatched_patents = [
        {
            "title": doc.get("title") or Path(doc.get("path") or "").name,
            "path": doc.get("path") or "",
        }
        for index, doc in enumerate(patent_docs)
        if index not in mapped_hit_patent_indexes
    ]
    matched_patent_count = min(
        len(patent_docs),
        len(mapped_hit_patent_indexes) + len(unresolved_hit_keys),
    )
    unmatched_patent_count = max(0, len(patent_docs) - matched_patent_count)
    if len(unmatched_patents) > unmatched_patent_count:
        unmatched_patents = unmatched_patents[:unmatched_patent_count]

    print(f"[info] matches total: {len(matches)}")
    print(f"[info] actionable matches: {len(actionable_matches)}")
    print(f"[info] weak matches: {len(weak_matches)}")
    print(
        "[info] matched patents "
        f"(max_similarity>={hit_similarity_threshold:.2f}): "
        f"{matched_patent_count} / {len(patent_docs)}"
    )
    if unresolved_hit_keys:
        print(
            "[warn] hit patent groups unresolved by uploaded file mapping: "
            f"{len(unresolved_hit_keys)}"
        )

    for group in patent_groups[:12]:
        print(
            "  - "
            f"{group['patent_title']} ({display_file(group['patent_file'])}) | "
            f"matches={group['match_count']} | "
            f"max_similarity={group['max_similarity']:.2f}"
        )

    result_kind = "PASS"
    detail = "no obvious similar patterns found against uploaded patents"
    exit_code = 0
    if actionable_matches:
        result_kind = "WARNING"
        detail = f"found {len(actionable_matches)} actionable patent-like matches"
        exit_code = 10
    elif weak_matches:
        result_kind = "WARNING"
        detail = f"found {len(weak_matches)} weak patent-like matches"
        exit_code = 11

    result = {
        "result": result_kind,
        "detail": detail,
        "summary": {
            "uploaded_patent_files": len(patent_docs),
            "read_failed_files": len(patent_errors),
            "files_analyzed": to_int(
                (scanner_payload.get("scan_metadata") or {}).get("files_analyzed"),
                0,
            ),
            "patterns": len(patterns),
            "matches": len(matches),
            "actionable_matches": len(actionable_matches),
            "weak_matches": len(weak_matches),
            "matched_patent_files": matched_patent_count,
            "unmatched_patent_files": unmatched_patent_count,
            "hit_similarity_threshold": hit_similarity_threshold,
        },
        "funnel": funnel,
        "skills": {
            "scanner": {
                "loaded": scanner_skill.loaded,
                "source": scanner_skill.source,
                "note": scanner_skill.note,
            },
            "validator": {
                "loaded": validator_skill.loaded,
                "source": validator_skill.source,
                "note": validator_skill.note,
            },
        },
        "patents": [{"title": d["title"], "path": d["path"]} for d in patent_docs],
        "patent_errors": patent_errors,
        "patterns": patterns[: options.max_patterns],
        "matches": matches[: options.max_matches],
        "patent_groups": patent_groups[:100],
        "unmatched_patents": unmatched_patents[:100],
    }

    write_output_if_requested(result, options.output_json)
    print(f"[result] {result_kind}: {detail}")
    return exit_code


def write_output_if_requested(payload: dict[str, Any], output_json: str | None) -> None:
    if not output_json:
        return
    output_path = Path(output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[info] wrote json output: {output_path}")
