"""LLM 백엔드 클라이언트.

두 종류를 쓴다:
  - NvidiaLLM : NVIDIA 클라우드(OpenAI 호환)의 minimax-m3 등. 공격 페이로드
    생성/개선용. 안전튜닝이 약해 재밍 페이로드를 실제로 만들어 준다.
  - ClaudeCLI : 로컬에 설치된 `claude` CLI(로컬 OAuth 인증). 판정/분석 등
    '분류' 작업용. Claude 는 페이로드 생성은 거부하지만 판정은 잘 한다.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Any, Dict, List, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def strip_reasoning(text: str) -> str:
    """Nemotron/추론 모델이 content 에 인라인한 <think>...</think> 사고과정을 제거."""
    if not text:
        return text
    text = _THINK_RE.sub("", text)
    if "<think>" in text and "</think>" not in text:  # 닫히지 않은 잔재
        text = text.split("<think>", 1)[0]
    return text.strip()


# ---------------------------------------------------------------------------
# OpenAI 호환 chat 클라이언트 — 공격자/방어자/판정 백엔드
#   기본은 NVIDIA(minimax/nemotron 등)지만, base_url/api_key 를 넘기면 임의의
#   OpenAI 호환 엔드포인트(OpenAI·Together·Groq·로컬 Ollama·커스텀 방어서버 등)로
#   동작한다. 방어자 역할에 임의 엔드포인트를 지정해 그 모델이 타깃을 연기하게 할 수 있다.
# ---------------------------------------------------------------------------
class NvidiaLLM:
    def __init__(self, cfg: Dict[str, Any], model_id: Optional[str] = None,
                 base_url: Optional[str] = None, api_key: Optional[str] = None,
                 api_key_env: Optional[str] = None):
        a = cfg.get("attacker", {})  # 연결 파라미터(토큰/타임아웃) 공용 템플릿
        self.base_url = base_url or a.get("base_url", "https://integrate.api.nvidia.com/v1")
        self.model = model_id or a.get("model", "minimaxai/minimax-m3")
        self.max_tokens = int(a.get("max_tokens", 1200))
        self.temperature = float(a.get("temperature", 1.0))
        self.top_p = float(a.get("top_p", 0.95))
        self.timeout = int(a.get("timeout_sec", 120))
        # 우선순위: 직접 지정 api_key > 지정된 env 이름 > 기본 NVIDIA_API_KEY env
        if api_key:
            self.api_key = api_key
        else:
            env_name = api_key_env or a.get("api_key_env", "NVIDIA_API_KEY")
            self.api_key = os.environ.get(env_name)
            if not self.api_key:
                raise RuntimeError(
                    "API 키가 없습니다 (env '%s'). .env 에 설정하거나 엔드포인트에 "
                    "api_key 를 직접 지정하세요." % env_name)

    @retry(reraise=True, stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=2, max=15),
           retry=retry_if_exception_type((requests.RequestException,)))
    def chat(self, system: str, user: str,
             temperature: Optional[float] = None) -> str:
        payload = {
            "model": self.model,
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        resp = requests.post(
            self.base_url.rstrip("/") + "/chat/completions",
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]
        text = (msg.get("content") or "").strip()
        if not text:
            # 일부 Nemotron 추론 모델은 최종답을 reasoning_content 로 분리한다.
            text = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
        return strip_reasoning(text)

    def classify(self, system: str, user: str) -> str:
        """판정용 통일 인터페이스 (chat 과 동일, 낮은 temperature)."""
        return self.chat(system, user, temperature=0.2)


# ---------------------------------------------------------------------------
# Claude CLI — 판정/분석기 (로컬 OAuth)
# ---------------------------------------------------------------------------
class ClaudeCLI:
    def __init__(self, cfg: Dict[str, Any]):
        j = cfg.get("judge", {})
        self.model = j.get("model", "opus")            # 'opus' | 'sonnet' | full id
        self.timeout = int(j.get("timeout_sec", 120))
        self.bin = j.get("bin", "claude")

    def classify(self, system: str, user: str) -> str:
        """system 프롬프트로 분류기 역할을 지정하고 user 를 stdin 으로 전달.
        `claude -p --output-format json` 의 result 필드(텍스트)를 반환.
        """
        cmd = [
            self.bin, "-p",
            "--model", self.model,
            "--output-format", "json",
            "--no-session-persistence",
            "--disallowedTools", "Bash Read Edit Write WebSearch WebFetch Task",
            "--system-prompt", system,
        ]
        try:
            proc = subprocess.run(
                cmd, input=user, capture_output=True, text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return ""
        if proc.returncode != 0:
            return ""
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return proc.stdout.strip()
        if data.get("is_error"):
            return ""
        return data.get("result", "") or ""

    def chat(self, system: str, user: str,
             temperature: Optional[float] = None) -> str:
        """공격자/방어자용 통일 인터페이스. Claude CLI 는 temperature 를 받지 않으므로 무시.
        참고: Claude 는 재밍 페이로드 '생성'은 거부하는 경향이 있어 공격자로 쓰면
        대부분 거부 응답을 반환한다(방어 강도와 무관)."""
        return self.classify(system, user)


# ---------------------------------------------------------------------------
# 공용: 모델 응답에서 첫 JSON 객체 추출
# ---------------------------------------------------------------------------
def extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {}
