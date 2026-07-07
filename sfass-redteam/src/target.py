"""HTTP 타깃 어댑터.

대회 엔드포인트의 실제 요청/응답 스키마를 알 수 없으므로, config 로 완전히
매핑 가능하게 만든다. body_template 안의 "{{PAYLOAD}}"/"{{SESSION_ID}}"
플레이스홀더를 치환해 전송하고, dotted-path 로 응답을 표준 필드로 뽑아낸다.
"""
from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


def dig(obj: Any, path: Optional[str]) -> Any:
    """dotted-path 로 중첩 dict/list 접근. 배열 인덱스(정수) 지원.

    예) dig(resp, "choices.0.message.content")
    경로가 없으면 None 반환.
    """
    if path is None or path == "":
        return None
    cur = obj
    for part in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def _substitute(template: Any, mapping: Dict[str, str]) -> Any:
    """dict/list/str 를 재귀 순회하며 {{KEY}} 플레이스홀더를 치환."""
    if isinstance(template, str):
        out = template
        for k, v in mapping.items():
            out = out.replace("{{" + k + "}}", v)
        return out
    if isinstance(template, dict):
        return {k: _substitute(v, mapping) for k, v in template.items()}
    if isinstance(template, list):
        return [_substitute(v, mapping) for v in template]
    return template


class TargetResponse:
    """타깃 1회 호출 결과를 표준화한 객체."""

    def __init__(self, ok: bool, status: int, raw: Any,
                 agent_text: str, action: Any, session_id: Optional[str],
                 error: Optional[str] = None):
        self.ok = ok                    # HTTP 레벨 성공 여부
        self.status = status
        self.raw = raw                  # 파싱된 JSON(or 원문 텍스트)
        self.agent_text = agent_text    # 에이전트 텍스트 응답
        self.action = action            # 에이전트가 택한 동작(있으면)
        self.session_id = session_id
        self.error = error

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok, "status": self.status,
            "agent_text": self.agent_text, "action": self.action,
            "session_id": self.session_id, "error": self.error,
            "raw": self.raw,
        }

    @staticmethod
    def flatten_str(raw: Any) -> str:
        """성공 정규식 매칭용으로 응답 전체를 문자열로."""
        if isinstance(raw, str):
            return raw
        try:
            return json.dumps(raw, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(raw)


class HTTPTarget:
    def __init__(self, cfg: Dict[str, Any]):
        self.t = cfg["target"]
        self.r = cfg["response"]
        self.session = requests.Session()
        self._server_session_id: Optional[str] = None

    def reset_session(self) -> None:
        self._server_session_id = None

    @retry(reraise=True,
           stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=1, max=8),
           retry=retry_if_exception_type(requests.RequestException))
    def _post(self, url: str, method: str, headers: Dict[str, str],
              body: Any, timeout: int) -> requests.Response:
        return self.session.request(
            method=method, url=url, headers=headers,
            json=body, timeout=timeout,
        )

    def send(self, payload: str) -> TargetResponse:
        mapping = {
            "PAYLOAD": payload,
            "SESSION_ID": self._server_session_id or "",
        }
        body = _substitute(copy.deepcopy(self.t["body_template"]), mapping)
        headers = _substitute(copy.deepcopy(self.t.get("headers", {})), mapping)

        try:
            resp = self._post(
                url=self.t["url"],
                method=self.t.get("method", "POST"),
                headers=headers,
                body=body,
                timeout=int(self.t.get("timeout_sec", 30)),
            )
        except requests.RequestException as e:
            return TargetResponse(False, -1, None, "", None, None, error=str(e))

        # 응답 파싱: JSON 우선, 실패 시 원문 텍스트
        try:
            raw = resp.json()
        except (ValueError, json.JSONDecodeError):
            raw = resp.text

        agent_text = self._extract_text(raw)
        action = dig(raw, self.r.get("action_path")) if isinstance(raw, (dict, list)) else None

        # 세션 관리
        sess_cfg = self.t.get("session", {}) or {}
        if sess_cfg.get("enabled"):
            sid = dig(raw, sess_cfg.get("id_field")) if isinstance(raw, (dict, list)) else None
            if sid:
                self._server_session_id = str(sid)

        return TargetResponse(
            ok=resp.ok, status=resp.status_code, raw=raw,
            agent_text=agent_text or "", action=action,
            session_id=self._server_session_id,
        )

    def _extract_text(self, raw: Any) -> str:
        if isinstance(raw, str):
            return raw
        val = dig(raw, self.r.get("agent_text_path"))
        if val is None:
            # 경로 매핑이 틀렸을 때를 대비해 흔한 후보를 폴백 탐색
            for cand in ("reply", "response", "message", "content", "text", "output", "answer"):
                val = dig(raw, cand)
                if isinstance(val, str) and val:
                    return val
            return json.dumps(raw, ensure_ascii=False)[:2000]
        return val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)

    @staticmethod
    def flatten_str(raw: Any) -> str:
        """성공 정규식 매칭용으로 응답 전체를 문자열로."""
        if isinstance(raw, str):
            return raw
        try:
            return json.dumps(raw, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(raw)
