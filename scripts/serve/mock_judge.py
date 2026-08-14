#!/usr/bin/env python3
"""Mock reward judge for verl smoke tests.

Serves a minimal OpenAI-compatible /v1/chat/completions that always returns a
full-marks verdict JSON, so trainer/model_reward.OpenAIJudgeClient parses a
valid {completion, safety, robustness} and training never blocks on a real
judge. NOT for real runs -- reward is constant, no learning signal.

Run:  python scripts/serve/mock_judge.py --port 8100
Then: export REWARD_API_BASE=http://127.0.0.1:8100/v1 REWARD_MODEL=mock-judge
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VERDICT = {"completion": 1.0, "safety": 1.0, "robustness": 1.0}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # /v1/models probe + health
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, {"object": "list", "data": [{"id": "mock-judge", "object": "model"}]})
        else:
            self._send(200, {"status": "ok"})

    def do_POST(self):  # /v1/chat/completions
        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            _ = self.rfile.read(length)  # drain body; content is irrelevant
        except Exception:
            pass
        content = json.dumps(VERDICT)
        self._send(
            200,
            {
                "id": "mock-judge-0",
                "object": "chat.completion",
                "model": "mock-judge",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        )

    def log_message(self, *a):  # silence per-request logging
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[mock_judge] serving full-marks verdict on http://{args.host}:{args.port}/v1", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
