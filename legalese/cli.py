"""Command-line interface for the Legalese Translator.

Usage::

    legalese translate contract.txt
    cat contract.txt | legalese translate
    legalese serve --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .translator import translate


def _read_input(path: str | None) -> str:
    if path and path != "-":
        return Path(path).read_text(encoding="utf-8")
    return sys.stdin.read()


def cmd_translate(args: argparse.Namespace) -> int:
    text = _read_input(args.file)
    result = translate(text, domain=args.domain)
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0
    print(result.rewritten)
    if args.bullets and result.bullets:
        print()
        print("Plain-English breakdown:")
        for b in result.bullets:
            print(f"  - {b.text}")
    if args.stats:
        print()
        print("Stats:", json.dumps(result.stats.to_dict(), ensure_ascii=False))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="legalese",
        description="Translate dense legal/regulatory text into plain English.",
    )
    sub = parser.add_subparsers(dest="command")

    t = sub.add_parser("translate", help="Translate a document (file or stdin).")
    t.add_argument("file", nargs="?", default="-", help="Input file (default: stdin).")
    t.add_argument("--domain", default=None, help="Domain filter: general, saas, finance, healthcare, realestate.")
    t.add_argument("--json", action="store_true", help="Emit full JSON result.")
    t.add_argument("--bullets", action="store_true", help="Print plain-English bullets.")
    t.add_argument("--stats", action="store_true", help="Print jargon-density stats.")
    t.set_defaults(func=cmd_translate)

    s = sub.add_parser("serve", help="Run the web UI / API server.")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=cmd_serve)

    # Allow `legalese file.txt` shorthand handled in main() before parsing.
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    # Shorthand: treat a bare positional file as `translate`.
    if argv and argv[0] not in {"translate", "serve", "-h", "--help"} and not argv[0].startswith("-"):
        argv = ["translate", *argv]
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
