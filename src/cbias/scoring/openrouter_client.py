"""OpenRouter API client helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

import requests


OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class OpenRouterClient:
	"""Minimal OpenRouter chat completion client."""

	api_key: str
	model: str
	timeout_seconds: int = 120
	http_referer: str = "https://github.com/stvsever/cbias-emotion-workflow"
	x_title: str = "cbias-emotion-workflow"
	_session: requests.Session = field(init=False, repr=False)

	def __post_init__(self) -> None:
		self._session = requests.Session()

	def chat_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> dict[str, Any]:
		"""Request a JSON response object from the configured model."""
		headers = {
			"Authorization": f"Bearer {self.api_key}",
			"Content-Type": "application/json",
			"HTTP-Referer": self.http_referer,
			"X-Title": self.x_title,
		}
		payload = {
			"model": self.model,
			"messages": [
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": user_prompt},
			],
			"temperature": temperature,
			"response_format": {"type": "json_object"},
		}
		response = self._session.post(
			OPENROUTER_CHAT_URL,
			headers=headers,
			json=payload,
			timeout=self.timeout_seconds,
		)
		response.raise_for_status()
		body = response.json()
		choices = body.get("choices")
		if not choices:
			raise RuntimeError("OpenRouter returned no choices")
		content = choices[0].get("message", {}).get("content", "")
		if not content:
			raise RuntimeError("OpenRouter returned empty content")
		return _extract_json_object(content)


def _extract_json_object(content: str) -> dict[str, Any]:
	"""Parse JSON, handling fenced and mixed-content outputs."""
	text = content.strip()
	if text.startswith("```"):
		parts = text.split("```")
		for part in parts:
			candidate = part.strip()
			if candidate.startswith("json"):
				candidate = candidate[4:].strip()
			if candidate.startswith("{") and candidate.endswith("}"):
				return json.loads(candidate)

	try:
		return json.loads(text)
	except json.JSONDecodeError:
		start = text.find("{")
		end = text.rfind("}")
		if start == -1 or end == -1 or end <= start:
			raise
		return json.loads(text[start : end + 1])
