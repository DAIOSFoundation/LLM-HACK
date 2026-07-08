"""헤디드 Playwright 로그인 캡처 — 실제 브라우저(Chrome→Edge→chromium) + 영속 프로필.

사용: python login_capture.py <url> <out_json> [profile_dir]

시크릿 창이 아니라 '실제 설치된 브라우저'를 영속 프로필로 띄운다. 사용자가 직접 로그인하면
세션 쿠키가 생기고(또는 창을 닫으면) 캡처해 out_json 에 {cookies, cookie_header, url} 저장.
영속 프로필이라 한 번 로그인하면 다음 캡처 때 로그인 상태가 유지된다.
"""
import json
import os
import sys
import time

SESSION_HINTS = ("session", "token", "auth", "sid", "jwt", "access", "csrf", "connect")


def _launch(p, profile_dir):
    """실제 Chrome → Edge → 번들 chromium 순으로 영속 컨텍스트 실행(headed)."""
    args = ["--no-first-run", "--no-default-browser-check"]
    for kw in ({"channel": "chrome"}, {"channel": "msedge"}, {}):
        try:
            return p.chromium.launch_persistent_context(
                profile_dir, headless=False, args=args, **kw)
        except Exception:
            continue
    return None


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: login_capture.py <url> <out_json> [profile_dir]", file=sys.stderr)
        return 2
    url, out = sys.argv[1], sys.argv[2]
    profile_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.expanduser("~/.cache/sfass-login-profile")
    os.makedirs(profile_dir, exist_ok=True)
    # 이전 실행이 강제종료되며 남긴 스테일 락 제거(재실행 실패 방지)
    for _f in ("SingletonLock", "SingletonCookie", "SingletonSocket"):
        try:
            os.remove(os.path.join(profile_dir, _f))
        except OSError:
            pass
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:  # noqa: BLE001
        print("playwright import 실패: %s" % e, file=sys.stderr)
        return 3

    with sync_playwright() as p:
        ctx = _launch(p, profile_dir)
        if ctx is None:
            print("브라우저 실행 실패(Chrome/Edge/chromium 모두 불가)", file=sys.stderr)
            return 4
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
        except Exception:
            pass
        try:
            baseline = {c["name"] for c in ctx.cookies()}
        except Exception:
            baseline = set()
        last = []
        deadline = time.time() + 300  # 사용자 로그인 최대 5분 대기
        while time.time() < deadline:
            try:
                last = ctx.cookies()
                cur = page.url
            except Exception:
                break  # 사용자가 브라우저를 닫음 → last 사용
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
        with open(out, "w", encoding="utf-8") as f:
            json.dump({
                "cookies": [{k: c.get(k) for k in ("name", "value", "domain", "path")} for c in last],
                "cookie_header": hdr, "url": final_url,
            }, f, ensure_ascii=False)
        try:
            ctx.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
