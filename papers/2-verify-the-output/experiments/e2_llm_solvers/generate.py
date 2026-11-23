"""Ask a language model to write solvers from SPEC.md (certsim environment).

Each sample is one independent request with its own seed. The raw reply, the
extracted program and the request settings are saved under
samples/<model>/<seed>/, so every solver can be traced to its request.

    uv run python generate.py [model] [first_seed] [n] [temperature]
"""

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLLAMA = "http://127.0.0.1:11434/api/chat"


def ask(model, prompt, seed, temperature, num_ctx=16384):
    body = {"model": model, "stream": False, "think": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"seed": seed, "temperature": temperature, "num_ctx": num_ctx}}
    req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return json.loads(resp.read())


def extract(reply):
    blocks = re.findall(r"```python\s*\n(.*?)```", reply, re.S)
    return max(blocks, key=len) if blocks else None


def main(model="qwen3.8:27b-mlx", first_seed=1, n=1, temperature=0.7):
    spec = (HERE / "SPEC.md").read_text()
    for seed in range(first_seed, first_seed + n):
        out = HERE / "samples" / model.replace(":", "_") / f"{seed:03d}"
        if (out / "reply.md").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        resp = ask(model, spec, seed, temperature)
        reply = resp["message"]["content"]
        code = extract(reply)
        (out / "reply.md").write_text(reply)
        if code:
            (out / "solver.py").write_text(code)
        meta = {"model": model, "seed": seed, "temperature": temperature, "think": False,
                "seconds": round(time.perf_counter() - t0, 1), "eval_count": resp.get("eval_count"),
                "has_code": code is not None}
        (out / "request.json").write_text(json.dumps(meta, indent=1))
        print(f"{model} seed {seed}: {meta['seconds']} s, {meta['eval_count']} tokens, code: {meta['has_code']}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "qwen3.8:27b-mlx", int(a[1]) if len(a) > 1 else 1,
         int(a[2]) if len(a) > 2 else 1, float(a[3]) if len(a) > 3 else 0.7)
