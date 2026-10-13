"""One entry point for download, source freeze, inference, verification and evaluation."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("download", "verify-model"):
        p = commands.add_parser(command)
        p.add_argument("--model-dir", type=Path, required=True)
    p = commands.add_parser("prepare", help="Freeze matching images/ and raw/ inputs, no reference answers")
    p.add_argument("--input-dir", type=Path, required=True)
    p = commands.add_parser("run", help="Run the complete source-image + MinerU Raw composition pipeline")
    for flag in ("manifest", "model-dir", "output"):
        p.add_argument("--" + flag, type=Path, required=True)
    p.add_argument("--device", choices=("mps", "cuda"), required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--retry-failed", action="store_true", help="Explicit retry in a new attempt directory, preserving the failed attempt")
    p = commands.add_parser("verify", help="Verify completed run output SHA bindings without a model")
    p.add_argument("--run", type=Path, required=True)
    p = commands.add_parser("evaluate", help="Plan by default; scoring requires --execute")
    for flag in ("run", "gold", "output"):
        p.add_argument("--" + flag, type=Path, required=True)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--evaluator-source", type=Path)
    p.add_argument("--docker-image", default="btsl-evaluator:0.2.0")
    p.add_argument("--dataset", choices=("official", "custom"), default="official")
    p.add_argument("--expected-pages", type=int, default=1651)
    p.add_argument("--execute", action="store_true")
    p = commands.add_parser("_call", help=argparse.SUPPRESS)
    p.add_argument("--request", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command in ("download", "verify-model"):
        from .model import download, verify_model
        result = download(args.model_dir) if args.command == "download" else {"verified_files": verify_model(args.model_dir)}
    elif args.command == "prepare":
        from .inputs import prepare
        result = prepare(args.input_dir)
    elif args.command == "run":
        from .pipeline import run
        if args.retry_failed and not args.resume:
            parser.error("--retry-failed requires --resume")
        result = run(args.manifest, args.model_dir, args.output, args.device, args.resume, args.retry_failed)
        result = {k: v for k, v in result.items() if k != "files"}
    elif args.command == "verify":
        from .pipeline import verify_ready
        result = verify_ready(args.run)
        result = {"status": "FILES_PASS", "files": len(result["files"]), "pages": result.get("pages")}
    elif args.command == "evaluate":
        from .evaluate import evaluate
        result = evaluate(args.run, args.gold, args.output, args.docker_image, args.dataset,
                          args.expected_pages, args.baseline, args.execute, args.evaluator_source)
    else:
        from .pipeline import child
        child(args.request)
        return
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
