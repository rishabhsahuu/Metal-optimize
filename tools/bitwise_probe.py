#!/usr/bin/env python3
"""Greedy outputs from a running Strata server, for a bitwise comparison. Usage:
   python tools/bitwise_probe.py PORT OUT.json

Sends the same two prompts with thinking off and temperature 0 and saves the
completion text of each, exactly as the server returns it.
"""
import json
import sys
import time
import urllib.request

port = int(sys.argv[1])
out_path = sys.argv[2]
base = f"http://127.0.0.1:{port}"

PROMPTS = [
    "Write one sentence about the ocean.",
    "List three colors, comma separated.",
]

t0 = time.time()
while time.time() - t0 < 300:
    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=3) as r:
            if r.status == 200:
                break
    except Exception:
        pass
    time.sleep(2)
else:
    sys.exit("server did not come up")

out = []
for prompt in PROMPTS:
    body = json.dumps({
        "model": "m",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 24,
        "temperature": 0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    text = d["choices"][0]["message"].get("content") or ""
    out.append({"prompt": prompt, "text": text, "usage": d.get("usage", {}),
                "wall_s": round(time.time() - t, 2)})
    print(prompt, "->", repr(text[:80]))

with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print("saved", out_path)
