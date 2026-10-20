"""Separate six-sample LLM reasoning check for the v2 references; does not
alter frozen accuracy. Key comes from stdin or JEV_API_KEY; never written."""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import sys
from pathlib import Path

from jev import _post

OUT = Path(__file__).resolve().parent / "outputs/frozen_v2/llm"
SYSTEM = (
    "You are assessing six preselected room-occupancy minutes from WiFi CSI "
    "and FMCW radar. Use only the supplied physical descriptors, training-basis "
    "PCA coordinates, training-class statistics/examples, and two fixed model "
    "probabilities and predictions. The input does not contain raw waveforms or "
    "images, so do not claim to have inspected them. For each ID return "
    "occupied_probability (0..1), predicted_class (empty or occupied), and a "
    "reason of at most 22 words naming specific CSI evidence, radar evidence, "
    "and the reference-model outputs. Explain conflicts or weak evidence "
    "honestly. Do not use sample ordering or infer prevalence. Return a JSON "
    "object with a results array."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("gpt-4o-mini", "gpt-5.4"), required=True)
    args = parser.parse_args()
    key = os.environ.get("JEV_API_KEY", "")
    if not key:
        key = getpass.getpass("API key: ") if sys.stdin.isatty() else sys.stdin.readline().strip()
    if not key:
        raise RuntimeError("API key required")
    request = json.loads((OUT / "explainability_request.json").read_text())
    payload = {"model": args.model, "response_format": {"type": "json_object"},
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": json.dumps(request, separators=(",", ":"))}],
               "max_completion_tokens": 5000}
    if args.model == "gpt-4o-mini":
        payload["temperature"] = 0.2
    data = _post("https://api.openai.com/v1/chat/completions",
                 {"Authorization": "Bearer " + key, "Content-Type": "application/json"}, payload)
    response = json.loads(data["choices"][0]["message"]["content"])
    received = [str(x["id"]) for x in response["results"]]
    expected = [str(x["id"]) for x in request["minutes"]]
    if len(received) != len(expected) or set(received) != set(expected):
        raise ValueError("Response sample IDs do not match request")
    (OUT / f"explainability_{args.model}.json").write_text(json.dumps({
        "model": args.model, "purpose": "qualitative explanations only; excluded from frozen accuracy",
        "system_prompt": SYSTEM,
        "request_sha256": hashlib.sha256(json.dumps(request).encode()).hexdigest(),
        "response": response, "usage": data.get("usage", {})}, indent=2))
    print(f"Saved {len(received)} explanations for {args.model}", flush=True)


if __name__ == "__main__":
    main()
