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
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import Any
from xml.etree import ElementTree


MAX_PATENT_FILES = 8
MAX_PATENT_TEXT_CHARS = 12000
MAX_PATTERNS = 30
MAX_MATCHES = 50
PATENT_HIT_SIMILARITY_THRESHOLD = 0.5


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


class PatentSkillAdapter:
    """Load scanner/validator skill markdown from local files with fallback."""

    DEFAULT_SKILL_ROOT = Path(__file__).resolve().parent / "skills"
    SCANNER_SKILL_REL = Path("code-patent-scanner") / "SKILL.md"
    VALIDATOR_SKILL_REL = Path("code-patent-validator") / "SKILL.md"
    MAX_SKILL_CHARS = 12000

    SCANNER_FALLBACK = """# Code Patent Scanner (Fallback)
Role: discover distinctive technical patterns from code.
Rules:
1. Analyze repository source files and identify high-value technical patterns.
2. Score each pattern by 4 dimensions:
   - distinctiveness (0-4)
   - sophistication (0-3)
   - system_impact (0-3)
   - frame_shift (0-3)
3. Only report patterns with total score >= 8.
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

    def _resolve_claude_cli_path(self) -> str | None:
        configured = (os.getenv("CLAUDE_CODE_CLI_PATH") or "").strip().strip('"').strip("'")
        if configured:
            return configured
        for candidate in ("claude", "claude.exe", "claude.cmd"):
            resolved = which(candidate)
            if resolved:
                return resolved
        return None

    def check_available(self) -> tuple[bool, str]:
        cli_path = self._resolve_claude_cli_path()
        if not cli_path:
            return False, "claude CLI not found in PATH and CLAUDE_CODE_CLI_PATH not set"
        try:
            completed = subprocess.run(
                [cli_path, "--version"],
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
    ) -> str:
        sdk = self._load_sdk()
        AssistantMessage = sdk["AssistantMessage"]
        ClaudeAgentOptions = sdk["ClaudeAgentOptions"]
        ClaudeSDKClient = sdk["ClaudeSDKClient"]
        ResultMessage = sdk["ResultMessage"]
        TextBlock = sdk["TextBlock"]

        cli_path = self._resolve_claude_cli_path()
        if not cli_path:
            raise RuntimeError("claude CLI not found")

        env: dict[str, str] = {}
        if self.anthropic_api_key and not self.anthropic_api_key.startswith("sk-ant-xxx"):
            env["ANTHROPIC_API_KEY"] = self.anthropic_api_key
        if self.anthropic_base_url:
            env["ANTHROPIC_BASE_URL"] = self.anthropic_base_url

        options = ClaudeAgentOptions(
            permission_mode="bypassPermissions",
            cwd=workspace_path.resolve(),
            model=self.model,
            system_prompt=system_prompt,
            env=env,
            # Include user settings so local Claude login/session can be reused.
            setting_sources=["user", "project"],
            cli_path=cli_path,
        )

        parts: list[str] = []
        sdk_client = ClaudeSDKClient(options=options)
        session_id = str(uuid.uuid4())

        try:
            async with asyncio.timeout(timeout):
                await sdk_client.connect()
                await sdk_client.query(message, session_id=session_id)
                async for sdk_message in sdk_client.receive_response():
                    if isinstance(sdk_message, AssistantMessage):
                        for block in sdk_message.content:
                            if isinstance(block, TextBlock) and (block.text or "").strip():
                                parts.append(block.text)
                    elif isinstance(sdk_message, ResultMessage):
                        result_text = (sdk_message.result or "").strip()
                        if result_text:
                            parts.append(result_text)
        finally:
            try:
                await sdk_client.disconnect()
            except Exception:
                pass

        output = "\n".join(part for part in parts if part.strip()).strip()
        if not output:
            raise RuntimeError("cloudcode returned empty response")
        return output

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

    try:
        content = path.read_text(encoding="utf-8", errors="ignore").strip()
        if content and "\x00" not in content:
            return content
    except Exception:
        return ""
    return ""


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
    return None


def normalize_patterns(raw_patterns: Any, *, max_patterns: int = MAX_PATTERNS) -> list[dict[str, Any]]:
    if not isinstance(raw_patterns, list):
        return []

    patterns: list[dict[str, Any]] = []
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

        if total >= 8:
            patterns.append(pattern)

    patterns.sort(key=lambda x: x.get("score_total", 0), reverse=True)
    return patterns[:max_patterns]


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

    patent_docs, patent_errors = load_patent_documents(
        patent_paths,
        max_patent_files=options.max_patent_files,
        max_patent_text_chars=options.max_patent_text_chars,
    )
    print(f"[info] patent docs loaded: {len(patent_docs)}")
    if patent_errors:
        print(f"[warn] patent doc errors: {len(patent_errors)}")
        for err in patent_errors[:8]:
            print(f"  - {err}")

    if not patent_docs:
        print("[error] all patent files failed to read", file=sys.stderr)
        return 4

    print("[info] scanner mode: Claude Code autonomous repository reading")
    scanner_prompt = build_scanner_prompt(scanner_skill, repo_path)
    print("[info] running code-patent-scanner ...")
    scanner_raw = await cloud_client.analyze(
        message=scanner_prompt,
        system_prompt="You are a strict code patent pattern analyzer. Reply with valid JSON object only.",
        workspace_path=repo_path,
        timeout=options.timeout,
    )
    scanner_payload = extract_json_payload(scanner_raw)
    if scanner_payload is None:
        print("[error] scanner output is not valid JSON object", file=sys.stderr)
        return 6

    patterns = normalize_patterns(scanner_payload.get("patterns"), max_patterns=options.max_patterns)
    print(f"[info] patterns extracted (score>=8): {len(patterns)}")
    for pattern in patterns[:8]:
        print(f"  - {pattern['pattern_id']}: {pattern['title']} (score={pattern['score_total']}/13)")

    if not patterns:
        result = {
            "result": "PASS",
            "detail": "no high-score technical patterns found",
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
            "patterns": [],
            "matches": [],
            "patent_groups": [],
            "patent_errors": patent_errors,
        }
        write_output_if_requested(result, options.output_json)
        print("[result] PASS: no high-score patterns to compare")
        return 0

    validator_prompt = build_validator_prompt(validator_skill, patterns, patent_docs)
    print("[info] running code-patent-validator ...")
    validator_raw = await cloud_client.analyze(
        message=validator_prompt,
        system_prompt="You are a strict patent pattern validator. Reply with valid JSON object only.",
        workspace_path=repo_path,
        timeout=options.timeout,
    )
    validator_payload = extract_json_payload(validator_raw)
    if validator_payload is None:
        print("[error] validator output is not valid JSON object", file=sys.stderr)
        return 7

    matches = normalize_matches(validator_payload.get("matches"), max_matches=options.max_matches)
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
        "patent_groups": patent_groups[: options.max_patent_files],
        "unmatched_patents": unmatched_patents[: options.max_patent_files],
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
