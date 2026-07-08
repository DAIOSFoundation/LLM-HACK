"""헤디드 Playwright 로그인 캡처.

사용: python login_capture.py <url> <out_json>
브라우저를 띄워 사용자가 직접 로그인하게 하고, 세션 쿠키가 생기면(또는 브라우저를 닫으면)
쿠키를 캡처해 out_json 에 {cookies, cookie_header, url} 로 저장한다.
"""
import json
import sys
import time

SESSION_HINTS = ("session", "token", "auth", "sid", "jwt", "access", "csrf", "connect")


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: login_capture.py <url> <out_json>", file=sys.stderr)
        return 2
    url, out = sys.argv[1], sys.argv[2]
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:  # noqa: BLE001
        print("playwright import 실패: %s" % e, file=sys.stderr)
        return 3

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        ctx = browser.new_context()
        page = ctx.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
        except Exception:
            pass
        try:
            baseline = {c["name"] for c in ctx.cookies()}
        except Exception:
            baseline = set()
        last = []
        deadline = time.time() + 300  # 최대 5분 대기(사용자 로그인)
        while time.time() < deadline:
            try:
                last = ctx.cookies()
                cur = page.url
            except Exception:
                break  # 사용자가 브라우저/컨텍스트를 닫음 → last 사용
            names = {c["name"] for c in last}
            new = names - baseline
            sessionish = any(any(h in c["name"].lower() for h in SESSION_HINTS) for c in last)
            if new and (sessionish or "login" not in (cur or "").lower()):
                break
            time.sleep(2)
        try:
            final_url = page.url
        except Exception:
            final_url = url
        hdr = "; ".join("%s=%s" % (c["name"], c["value"]) for c in last)
        data = {
            "cookies": [{k: c.get(k) for k in ("name", "value", "domain", "path")} for c in last],
            "cookie_header": hdr, "url": final_url,
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        try:
            browser.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
