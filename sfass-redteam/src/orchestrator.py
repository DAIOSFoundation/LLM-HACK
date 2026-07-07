"""오케스트레이터 — 자동 공격 루프.

목표(금지 동작)별로:
  1) (옵션) 정찰로 시스템 프롬프트/동작목록/툴포맷 유출
  2) 라운드마다 여러 전략으로 후보 페이로드 생성 → 타깃 전송 → 판정
  3) 실패 시 로봇 응답을 근거로 refine → 재전송
  4) 성공 시 페이로드 뱅크에 저장하고 다음 목표로

전략은 라운드마다 순환 선택하여 다양한 각도를 빠르게 훑는다.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from attacker import Attacker
from judge import Judge
from logger import RunLogger
from recon import Recon, recon_to_context
from strategies import STRATEGIES, Strategy
from target import HTTPTarget


class Orchestrator:
    def __init__(self, cfg: Dict[str, Any], target: HTTPTarget,
                 attacker: Attacker, judge: Judge, recon: Optional[Recon],
                 logger: RunLogger, console=None):
        self.cfg = cfg
        self.loop = cfg["loop"]
        self.target = target
        self.attacker = attacker
        self.judge = judge
        self.recon = recon
        self.logger = logger
        self.console = console

    def _say(self, msg: str) -> None:
        if self.console:
            self.console.print(msg)
        else:
            print(msg)

    def run(self, objectives: List[Dict[str, Any]]) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        recon_ctx: Optional[str] = None

        if self.loop.get("run_recon_first") and self.recon is not None:
            self._say("[bold cyan]● 정찰(recon) 시작[/]" if self.console else "● 정찰 시작")
            summary = self.recon.run(log_fn=self.logger.event)
            recon_ctx = recon_to_context(summary)
            self.logger.event({"phase": "recon_summary", "summary": summary})
            self._say("[green]정찰 완료[/]" if self.console else "정찰 완료")
            if recon_ctx:
                self._say(recon_ctx)

        for obj in objectives:
            res = self._attack_objective(obj, recon_ctx)
            results[obj["id"]] = res
        return results

    def _attack_objective(self, objective: Dict[str, Any],
                          recon_ctx: Optional[str]) -> Dict[str, Any]:
        oid = objective["id"]
        self._say(("\n[bold yellow]▶ 목표: %s[/] — %s" % (oid, objective["description"]))
                  if self.console else "\n▶ 목표: %s — %s" % (oid, objective["description"]))

        max_attempts = int(self.loop.get("max_attempts_per_objective", 25))
        cands = int(self.loop.get("candidates_per_round", 3))
        delay = float(self.loop.get("request_delay_sec", 1.0))
        stop_first = bool(self.loop.get("stop_on_first_success", True))

        history: List[Dict[str, Any]] = []
        attempts = 0
        successes: List[Dict[str, Any]] = []
        strat_idx = 0

        while attempts < max_attempts:
            # 이번 라운드에 사용할 전략들(순환)
            round_strats: List[Strategy] = []
            for _ in range(cands):
                round_strats.append(STRATEGIES[strat_idx % len(STRATEGIES)])
                strat_idx += 1

            for strat in round_strats:
                if attempts >= max_attempts:
                    break
                attempts += 1

                # 같은 전략의 직전 실패가 있으면 refine, 아니면 generate
                prev = _last_for_strategy(history, strat.key)
                if prev and prev.get("agent_text"):
                    gen = self.attacker.refine(
                        objective, strat, prev["payload"], prev["agent_text"],
                        recon_ctx, history)
                else:
                    gen = self.attacker.generate(objective, strat, recon_ctx, history)

                payload = gen.get("payload", "")
                if self.loop.get("reset_session_each_attempt"):
                    self.target.reset_session()

                resp = self.target.send(payload)
                verdict = self.judge.judge(objective, resp)

                record = {
                    "phase": "attempt", "objective": oid, "attempt": attempts,
                    "strategy": strat.key, "strategy_name": strat.name,
                    "rationale": gen.get("rationale") or gen.get("change"),
                    "payload": payload,
                    "agent_text": resp.agent_text, "action": resp.action,
                    "http_status": resp.status, "http_ok": resp.ok, "error": resp.error,
                    "success": verdict["success"], "judge_source": verdict["source"],
                    "confidence": verdict["confidence"], "evidence": verdict["evidence"],
                }
                history.append(record)
                self.logger.event(record)
                self._print_attempt(record)

                if verdict["success"]:
                    successes.append(record)
                    self.logger.bank_success({
                        "objective": oid, "strategy": strat.key,
                        "payload": payload, "evidence": verdict["evidence"],
                        "judge_source": verdict["source"],
                    })
                    self._say("[bold green]✔ 성공![/] [%s] via %s"
                              % (oid, strat.name) if self.console
                              else "✔ 성공! [%s] via %s" % (oid, strat.name))
                    if stop_first:
                        return {"success": True, "attempts": attempts,
                                "winner": record, "successes": successes}

                time.sleep(delay)

        return {"success": len(successes) > 0, "attempts": attempts,
                "successes": successes}

    def _print_attempt(self, r: Dict[str, Any]) -> None:
        mark = "✔" if r["success"] else "·"
        head = "  %s #%d [%s] (%s, conf=%.2f)" % (
            mark, r["attempt"], r["strategy"], r["judge_source"], r["confidence"])
        if self.console:
            color = "green" if r["success"] else "white"
            self.console.print("[%s]%s[/]" % (color, head))
            if self.cfg.get("logging", {}).get("verbose"):
                self.console.print("     payload: %s" % (r["payload"] or "")[:160].replace("\n", " ⏎ "))
                self.console.print("     resp   : %s" % (r["agent_text"] or "")[:160].replace("\n", " ⏎ "))
        else:
            print(head)


def _last_for_strategy(history: List[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    for h in reversed(history):
        if h.get("strategy") == key:
            return h
    return None
