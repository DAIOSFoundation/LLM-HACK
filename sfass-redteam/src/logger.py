"""실행 로깅 + 성공 페이로드 뱅크."""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional


class RunLogger:
    def __init__(self, cfg: Dict[str, Any]):
        lg = cfg.get("logging", {})
        self.dir = lg.get("dir", "runs")
        self.bank_path = lg.get("payload_bank", "data/successful_payloads.jsonl")
        self.verbose = bool(lg.get("verbose", True))
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.bank_path) or ".", exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_path = os.path.join(self.dir, "run_%s.jsonl" % stamp)
        self._f = open(self.run_path, "a", encoding="utf-8")

    def event(self, record: Dict[str, Any]) -> None:
        record = dict(record)
        record.setdefault("ts", datetime.now().isoformat(timespec="seconds"))
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._f.flush()

    def bank_success(self, record: Dict[str, Any]) -> None:
        with open(self.bank_path, "a", encoding="utf-8") as bf:
            rec = dict(record)
            rec.setdefault("ts", datetime.now().isoformat(timespec="seconds"))
            bf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass
