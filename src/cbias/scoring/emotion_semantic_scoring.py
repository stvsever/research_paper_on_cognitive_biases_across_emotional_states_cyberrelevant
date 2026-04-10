"""High-resolution binary semantic scoring for each emotion term."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any

from dotenv import load_dotenv

from cbias.emotion.calibrate_dimensions import read_emotion_lexicon
from cbias.scoring.openrouter_client import OpenRouterClient
from cbias.scoring.semantic_dimensions import DIMENSIONS, BinarySemanticDimension


DEFAULT_OUTPUT_DIR = Path("src/review_stages/12_emotion_dimension_calibration/outputs")
DEFAULT_MODEL = "google/gemini-2.0-flash-001"
DEFAULT_FALLBACK_MODEL = "mistralai/mistral-small-3.1-24b-instruct"


@dataclass
class ScoringResult:
	"""Structured scoring result for one emotion."""

	emotion: str
	model_used: str
	rationale: str
	status: str
	error: str
	dimension_payload: dict[str, dict[str, Any]]
	original_payload: dict[str, Any]


def run_emotion_semantic_scoring(
	*,
	env_file: Path,
	emotion_file: Path,
	output_dir: Path,
	model: str = DEFAULT_MODEL,
	fallback_model: str | None = DEFAULT_FALLBACK_MODEL,
	max_workers: int = 50,
	retries: int = 2,
	timeout_seconds: int = 120,
	temperature: float = 0.0,
	limit: int | None = None,
	dry_run: bool = False,
) -> dict[str, Any]:
	"""Score each emotion via OpenRouter with high-resolution binary dimensions."""
	load_dotenv(dotenv_path=env_file, override=False)
	model = os.getenv("OPENROUTER_MODEL", model)
	fallback_model = os.getenv("OPENROUTER_FALLBACK_MODEL", fallback_model or "") or None
	max_workers = _parse_int_env("OPENROUTER_MAX_WORKERS", max_workers)
	emotions = read_emotion_lexicon(emotion_file)
	if limit is not None:
		emotions = emotions[: max(limit, 0)]

	output_dir.mkdir(parents=True, exist_ok=True)
	if not emotions:
		raise RuntimeError("No emotions found to score")

	api_key = os.getenv("OPENROUTER_API_KEY", "")
	if not dry_run and not api_key:
		raise RuntimeError("OPENROUTER_API_KEY is missing in environment")

	primary_client = None
	fallback_client = None
	if not dry_run:
		primary_client = OpenRouterClient(
			api_key=api_key,
			model=model,
			timeout_seconds=timeout_seconds,
		)
		if fallback_model:
			fallback_client = OpenRouterClient(
				api_key=api_key,
				model=fallback_model,
				timeout_seconds=timeout_seconds,
			)

	results: list[ScoringResult] = []
	with ThreadPoolExecutor(max_workers=max_workers) as executor:
		future_map = {
			executor.submit(
				_score_single_emotion,
				emotion,
				primary_client,
				fallback_client,
				retries,
				temperature,
				dry_run,
			): emotion
			for emotion in emotions
		}

		for future in as_completed(future_map):
			results.append(future.result())

	results.sort(key=lambda item: item.emotion.lower())
	wide_rows = _build_wide_rows(results)
	long_rows = _build_long_rows(results)
	_write_csv(output_dir / "emotion_semantic_scores_wide.csv", wide_rows)
	_write_csv(output_dir / "emotion_semantic_scores_long.csv", long_rows)
	_write_jsonl(output_dir / "emotion_semantic_scores_raw.jsonl", results)

	success_count = sum(1 for item in results if item.status == "ok")
	return {
		"total": len(results),
		"success": success_count,
		"failed": len(results) - success_count,
		"model": model,
		"fallback_model": fallback_model,
		"output_dir": str(output_dir),
		"dry_run": dry_run,
	}


def _score_single_emotion(
	emotion: str,
	primary_client: OpenRouterClient | None,
	fallback_client: OpenRouterClient | None,
	retries: int,
	temperature: float,
	dry_run: bool,
) -> ScoringResult:
	if dry_run:
		payload = _build_dry_run_payload(emotion)
		return _normalize_result(
			emotion=emotion,
			payload=payload,
			model_used="dry-run",
			status="ok",
			error="",
		)

	system_prompt = _build_system_prompt()
	user_prompt = _build_user_prompt(emotion)

	try:
		payload = _request_with_retry(
			client=primary_client,
			system_prompt=system_prompt,
			user_prompt=user_prompt,
			temperature=temperature,
			retries=retries,
		)
		return _normalize_result(
			emotion=emotion,
			payload=payload,
			model_used=primary_client.model if primary_client else "unknown",
			status="ok",
			error="",
		)
	except Exception as primary_error:
		if not fallback_client:
			return _failure_result(emotion, str(primary_error))

		try:
			payload = _request_with_retry(
				client=fallback_client,
				system_prompt=system_prompt,
				user_prompt=user_prompt,
				temperature=temperature,
				retries=retries,
			)
			return _normalize_result(
				emotion=emotion,
				payload=payload,
				model_used=fallback_client.model,
				status="ok",
				error=f"primary_failed={primary_error}",
			)
		except Exception as fallback_error:
			combined_error = f"primary={primary_error}; fallback={fallback_error}"
			return _failure_result(emotion, combined_error)


def _request_with_retry(
	*,
	client: OpenRouterClient | None,
	system_prompt: str,
	user_prompt: str,
	temperature: float,
	retries: int,
) -> dict[str, Any]:
	if client is None:
		raise RuntimeError("Client is not configured")

	last_error: Exception | None = None
	for attempt in range(retries + 1):
		try:
			return client.chat_json(
				system_prompt=system_prompt,
				user_prompt=user_prompt,
				temperature=temperature,
			)
		except Exception as error:  # noqa: PERF203 - explicit retry logic
			last_error = error
			if attempt >= retries:
				break
			sleep_seconds = (2**attempt) + random.random() * 0.3
			time.sleep(sleep_seconds)

	raise RuntimeError(f"OpenRouter request failed after retries: {last_error}")


def _build_system_prompt() -> str:
	dimension_lines = []
	for dim in DIMENSIONS:
		dimension_lines.append(
			f"- {dim.code}: {dim.name} ({dim.left_pole} <-> {dim.right_pole}). {dim.description}"
		)

	return "\n".join(
		[
			"You are an affective semantics rater for preregistered scientific workflow.",
			"Return strict JSON only.",
			"Use high-resolution binary semantics for each dimension.",
			"For each dimension, provide right_pole_score_0_1000 as integer in [0,1000], and confidence_0_1 in [0,1].",
			"Dimension definitions:",
			*dimension_lines,
			"Do not include markdown or extra keys outside schema.",
		]
	)


def _build_user_prompt(emotion: str) -> str:
	score_schema = {
		dim.code: {
			"right_pole_score_0_1000": 0,
			"confidence_0_1": 0.0,
			"note": "short evidence sentence",
		}
		for dim in DIMENSIONS
	}
	requested_schema = {
		"emotion": emotion,
		"scores": score_schema,
		"rationale": "<=30 words summary",
	}
	return "\n".join(
		[
			f"Emotion term: {emotion}",
			"Return JSON matching this schema exactly:",
			json.dumps(requested_schema, ensure_ascii=True),
		]
	)


def _build_dry_run_payload(emotion: str) -> dict[str, Any]:
	scores: dict[str, dict[str, Any]] = {}
	for dim in DIMENSIONS:
		base = _hash_to_int(f"{emotion}:{dim.code}")
		right = base % 1001
		confidence = round(((base // 1001) % 1000) / 1000.0, 3)
		scores[dim.code] = {
			"right_pole_score_0_1000": right,
			"confidence_0_1": confidence,
			"note": "dry-run synthetic value",
		}
	return {
		"emotion": emotion,
		"scores": scores,
		"rationale": "dry-run synthetic payload",
	}


def _hash_to_int(value: str) -> int:
	digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
	return int(digest[:12], 16)


def _normalize_result(
	*,
	emotion: str,
	payload: dict[str, Any],
	model_used: str,
	status: str,
	error: str,
) -> ScoringResult:
	score_blob = payload.get("scores", {}) if isinstance(payload, dict) else {}
	by_dimension: dict[str, dict[str, Any]] = {}

	for dim in DIMENSIONS:
		raw = _pick_dimension_blob(score_blob, dim)
		right = _clamp_int(raw.get("right_pole_score_0_1000"), 0, 1000)
		if right is None:
			right = _clamp_int(raw.get("score_0_1000"), 0, 1000)
		if right is None:
			right = 500

		confidence = _clamp_float(raw.get("confidence_0_1"), 0.0, 1.0)
		if confidence is None:
			confidence = 0.5

		signed = (2 * right) - 1000
		direction_label, direction_code = _direction_encoding(signed)
		by_dimension[dim.code] = {
			"dimension_name": dim.name,
			"left_pole": dim.left_pole,
			"right_pole": dim.right_pole,
			"left_pole_score_0_1000": 1000 - right,
			"right_pole_score_0_1000": right,
			"signed_score_m1000_to_p1000": signed,
			"direction_label": direction_label,
			"direction_code": direction_code,
			"confidence_0_1": confidence,
			"note": str(raw.get("note", "")),
		}

	rationale = str(payload.get("rationale", "")).strip() if isinstance(payload, dict) else ""
	return ScoringResult(
		emotion=emotion,
		model_used=model_used,
		rationale=rationale,
		status=status,
		error=error,
		dimension_payload=by_dimension,
		original_payload=payload if isinstance(payload, dict) else {},
	)


def _pick_dimension_blob(score_blob: Any, dim: BinarySemanticDimension) -> dict[str, Any]:
	if not isinstance(score_blob, dict):
		return {}

	keys_to_try = (
		dim.code,
		dim.name,
		dim.name.lower(),
		dim.name.upper(),
	)
	for key in keys_to_try:
		value = score_blob.get(key)
		if isinstance(value, dict):
			return value
	return {}


def _direction_encoding(signed_score: int) -> tuple[str, int]:
	if signed_score >= 50:
		return "toward_right_pole", 1
	if signed_score <= -50:
		return "toward_left_pole", -1
	return "balanced", 0


def _failure_result(emotion: str, error: str) -> ScoringResult:
	empty = {
		dim.code: {
			"dimension_name": dim.name,
			"left_pole": dim.left_pole,
			"right_pole": dim.right_pole,
			"left_pole_score_0_1000": 500,
			"right_pole_score_0_1000": 500,
			"signed_score_m1000_to_p1000": 0,
			"direction_label": "balanced",
			"direction_code": 0,
			"confidence_0_1": 0.0,
			"note": "",
		}
		for dim in DIMENSIONS
	}
	return ScoringResult(
		emotion=emotion,
		model_used="none",
		rationale="",
		status="error",
		error=error,
		dimension_payload=empty,
		original_payload={},
	)


def _clamp_int(value: Any, min_value: int, max_value: int) -> int | None:
	try:
		parsed = int(value)
	except (TypeError, ValueError):
		return None
	return max(min_value, min(max_value, parsed))


def _clamp_float(value: Any, min_value: float, max_value: float) -> float | None:
	try:
		parsed = float(value)
	except (TypeError, ValueError):
		return None
	return max(min_value, min(max_value, parsed))


def _parse_int_env(name: str, default: int) -> int:
	raw = os.getenv(name)
	if raw is None:
		return default
	try:
		return int(raw)
	except ValueError:
		return default


def _build_wide_rows(results: list[ScoringResult]) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	for item in results:
		row: dict[str, Any] = {
			"emotion": item.emotion,
			"model_used": item.model_used,
			"status": item.status,
			"error": item.error,
			"rationale": item.rationale,
		}
		for dim in DIMENSIONS:
			d = item.dimension_payload[dim.code]
			row[f"{dim.code}_right_pole_score_0_1000"] = d["right_pole_score_0_1000"]
			row[f"{dim.code}_left_pole_score_0_1000"] = d["left_pole_score_0_1000"]
			row[f"{dim.code}_signed_score_m1000_to_p1000"] = d["signed_score_m1000_to_p1000"]
			row[f"{dim.code}_direction_label"] = d["direction_label"]
			row[f"{dim.code}_direction_code"] = d["direction_code"]
			row[f"{dim.code}_confidence_0_1"] = d["confidence_0_1"]
		rows.append(row)
	return rows


def _build_long_rows(results: list[ScoringResult]) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	for item in results:
		for dim in DIMENSIONS:
			d = item.dimension_payload[dim.code]
			rows.append(
				{
					"emotion": item.emotion,
					"dimension_code": dim.code,
					"dimension_name": d["dimension_name"],
					"left_pole": d["left_pole"],
					"right_pole": d["right_pole"],
					"left_pole_score_0_1000": d["left_pole_score_0_1000"],
					"right_pole_score_0_1000": d["right_pole_score_0_1000"],
					"signed_score_m1000_to_p1000": d["signed_score_m1000_to_p1000"],
					"direction_label": d["direction_label"],
					"direction_code": d["direction_code"],
					"confidence_0_1": d["confidence_0_1"],
					"note": d["note"],
					"model_used": item.model_used,
					"status": item.status,
					"error": item.error,
				}
			)
	return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
	if not rows:
		return
	path.parent.mkdir(parents=True, exist_ok=True)
	fieldnames = list(rows[0].keys())
	with path.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def _write_jsonl(path: Path, results: list[ScoringResult]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		for item in results:
			record = {
				"emotion": item.emotion,
				"model_used": item.model_used,
				"status": item.status,
				"error": item.error,
				"rationale": item.rationale,
				"scores": item.dimension_payload,
				"original_payload": item.original_payload,
			}
			handle.write(json.dumps(record, ensure_ascii=True) + "\n")
