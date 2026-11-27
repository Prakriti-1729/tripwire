"""Terminal demo.

  python demo/cli.py                        # interactive
  python demo/cli.py -p "prompt" -p "..."   # non-interactive
  python demo/cli.py -p "..." --json        # one JSON line per prompt (used by the smoke test)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tripwire.common import DEFAULT_MODEL  # noqa: E402
from tripwire.serve import TripwireServer  # noqa: E402


def run(srv, prompt, a):
    text, reason, v = "", "", None
    if not a.json:
        print(f"\n> {prompt}")
    for kind, val in srv.respond(prompt, max_new_tokens=a.max_new_tokens, per_token=a.per_token):
        if kind == "verdict":
            v = val
            if not a.json:
                tag = "TRIGGERED" if v.fired else "ok"
                print(f"[tripwire] score {v.score:+.2f} (threshold {v.threshold:+.2f}, layer {v.layer}) -> {tag} "
                      f"[{v.prefill_ms:.0f} ms prefill]")
        elif kind == "text":
            if not a.json:
                print(val[len(text):], end="", flush=True)
            text = val
        else:
            reason = val
    if a.json:
        print(json.dumps({"prompt": prompt, "score": v.score, "fired": v.fired, "layer": v.layer,
                          "threshold": v.threshold, "n_chars": len(text), "stop": reason}), flush=True)
    else:
        print(f"\n[{reason}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("-p", "--prompt", action="append")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--per-token", action="store_true", help="also run the probe on each generated token")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    srv = TripwireServer(a.model)
    if a.prompt:
        for p in a.prompt:
            run(srv, p, a)
        return
    print("Tripwire CLI - empty line or Ctrl-D to quit")
    while True:
        try:
            p = input("\nprompt> ").strip()
        except EOFError:
            break
        if not p:
            break
        run(srv, p, a)


if __name__ == "__main__":
    main()
