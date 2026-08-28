#!/usr/bin/env python3
"""
Batch-run prompts against an Ollama /api/generate endpoint.

Reads an input file where each non-blank line is a prompt, sends each to the
model, and prints (or writes) the .response field — the Python equivalent of:

    curl -s http://HOST:11434/api/generate -d '{...}' | jq -r .response

Example:
    python3 ollama_batch.py prompts.txt
    python3 ollama_batch.py prompts.txt -m qwen2.5:7b-instruct -t 0.3 -o out.txt
"""

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request


class GenerationError(Exception):
    """Any failure that should trigger a retry (network, bad JSON, empty reply)."""


def generate(host: str, model: str, prompt: str, temperature: float,
             timeout: float) -> str:
    """POST one prompt to /api/generate and return the response text.

    Raises GenerationError on transport failure, unparseable output, or an
    empty/whitespace-only response — all of which are treated as retryable.
    """
    url = f"http://{host}/api/generate"
    payload = json.dumps({
        "model": model,
        "stream": False,
        "options": {"temperature": temperature},
        "prompt": prompt,
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as e:
        raise GenerationError(f"request failed: {e}") from e
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise GenerationError(f"bad response: {e}") from e

    response = data.get("response", "")
    #print(f"\nORIGINAL RESPONSE: {response}\n")
    if not response.strip():
        raise GenerationError("empty response")
    return response


def generate_with_retry(host: str, model: str, prompt: str, temperature: float,
                        timeout: float, retries: int, backoff: float,
                        max_backoff: float, index: int) -> str:
    """Call generate(), retrying up to `retries` times with exponential backoff.

    Returns the response on success, or "" if every attempt failed.
    """
    for attempt in range(1, retries + 1):
        try:
            return generate(host, model, prompt, temperature, timeout)
        except GenerationError as e:
            if attempt == retries:
                print(f"error: prompt {index} gave up after {retries} attempts "
                      f"({e})", file=sys.stderr)
                return ""
            # Exponential backoff with full jitter, capped at max_backoff.
            delay = min(backoff * (2 ** (attempt - 1)), max_backoff)
            delay = random.uniform(0, delay)
            print(f"warn: prompt {index} attempt {attempt}/{retries} failed "
                  f"({e}); retrying in {delay:.1f}s", file=sys.stderr)
            time.sleep(delay)
    return ""  # unreachable, but keeps the type checker happy


def main() -> int:
    p = argparse.ArgumentParser(description="Batch Ollama prompt runner.")
    p.add_argument("input", help="Input file; one prompt per line.")
    p.add_argument("-H", "--host", default="192.168.178.210:11434",
                   help="Ollama host:port (default: 192.168.178.210:11434).")
    p.add_argument("-m", "--model", default="gpt-3.5-turbo-0125",
                   help="Model name / alias (default: gpt-3.5-turbo-0125).")
    p.add_argument("-t", "--temperature", type=float, default=0.3,
                   help="Sampling temperature (default: 0.3).")
    p.add_argument("--timeout", type=float, default=300.0,
                   help="Per-request timeout in seconds (default: 300).")
    p.add_argument("-n", "--retries", type=int, default=5,
                   help="Max attempts per prompt before giving up (default: 5).")
    p.add_argument("--backoff", type=float, default=1.0,
                   help="Base backoff in seconds; doubles each retry (default: 1).")
    p.add_argument("--max-backoff", type=float, default=30.0,
                   help="Cap on backoff delay in seconds (default: 30).")
    p.add_argument("-o", "--output",
                   help="Write responses here instead of stdout.")
    p.add_argument("--sep", default="\n---\n",
                   help="Separator printed between responses (default: '---').")
    args = p.parse_args()

    # Read and filter prompts (skip blank lines).
    try:
        with open(args.input, encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
    except OSError as e:
        print(f"error: cannot read {args.input}: {e}", file=sys.stderr)
        return 1

    if not prompts:
        print("error: no prompts found in input file", file=sys.stderr)
        return 1

    out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    try:
        for i, prompt in enumerate(prompts, 1):
            print(f"\n[{i}/{len(prompts)}] {prompt}", file=sys.stderr)
            response = generate_with_retry(
                args.host, args.model, prompt, args.temperature, args.timeout,
                args.retries, args.backoff, args.max_backoff, i)

            #if i > 1:
            #    out.write(args.sep)
            out.write(response)
            out.flush()
        out.write("\n")
    finally:
        if args.output:
            out.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())