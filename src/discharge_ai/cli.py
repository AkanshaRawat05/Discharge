"""
cli.py
======

Command-line entry point — useful for smoke-testing without any UI.

    python -m discharge_ai.cli doctor                 # environment check
    python -m discharge_ai.cli scan                   # what documents exist
    python -m discharge_ai.cli run P1019              # extract → normalise → validate
    python -m discharge_ai.cli run --all              # every patient
    python -m discharge_ai.cli run P1019 --summary    # …and also generate the summary
    python -m discharge_ai.cli run P1022 --summary --force  # override a blocked case
    python -m discharge_ai.cli ask "what meds for P1019?"
    python -m discharge_ai.cli index                  # rebuild the FAISS index
    python -m discharge_ai.cli mcp                    # list MCP tools/resources/prompts
    python -m discharge_ai.cli agents                 # A2A AgentCard discovery
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from .llm.provider import provider_banner
from .observability import tracing
from .settings import configure_logging, settings


# --------------------------------------------------------------------------- #
def _print_header(title: str) -> None:
    print(f"\n{'=' * 78}\n  {title}\n{'=' * 78}")


def _aws_credentials_available() -> bool:
    """True when boto3's default credential chain resolves something.

    Never reads or prints the credentials themselves — only whether the chain
    (env vars / shared config / AWS_PROFILE / IAM role) found any.
    """
    try:
        import boto3

        return boto3.Session().get_credentials() is not None
    except Exception:  # noqa: BLE001 — the doctor command must never crash
        return False


# --------------------------------------------------------------------------- #
def cmd_doctor(_args: argparse.Namespace) -> int:
    """Check that the environment is wired up correctly."""
    import socket

    _print_header("Environment check")
    print(f"  python              {sys.version.split()[0]}")
    print(f"  project root        {settings.root}")
    print(f"  LLM                 {provider_banner()}")
    print(f"  AWS credentials     {'resolved' if _aws_credentials_available() else 'NOT FOUND'}"
          f"  (region {settings.aws_region})")
    print(f"  LangFuse            {'connected' if tracing.langfuse_available() else 'disabled'}")
    print(f"  rules.yaml          {settings.rules_path.exists()}")
    print(f"  prompts.yaml        {settings.prompts_path.exists()}")
    print(f"  input root          {settings.path('input_root')}")

    _print_header("Ports")
    ports = [
        ("Mock EHR", 8050), ("MCP Clinical Tools", 8200), ("MCP Analytics", 8201),
        ("Extractor", 8100), ("Validator", 8101), ("Normalizer", 8102),
        ("Monitor", 8103), ("Summary Generator", 8104), ("RAG Q&A", 8105),
        ("Host Orchestrator", 8083), ("HITL Dashboard", 8501),
    ]
    for label, port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.3)
            up = sock.connect_ex(("127.0.0.1", port)) == 0
        print(f"  {'🟢' if up else '⚪'} {label:<22} :{port}")

    _print_header("Documents")
    from .common.doc_loader import scan_incoming

    catalogue = scan_incoming()
    for patient_id, documents in catalogue.items():
        types = ", ".join(sorted(documents))
        print(f"  {patient_id}  {types}")
    print(f"\n  {len(catalogue)} patient(s) found.")

    if not _aws_credentials_available() and not settings.offline_mode:
        print(
            "\n  ! No AWS credentials could be resolved. Run `aws configure` (or "
            "export AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, or attach an IAM "
            "role), or set OFFLINE_MODE=1 for deterministic-only runs."
        )
        return 1
    return 0


def cmd_scan(_args: argparse.Namespace) -> int:
    from .common.doc_loader import scan_incoming

    _print_header("Incoming documents")
    catalogue = scan_incoming()
    for patient_id, documents in catalogue.items():
        print(f"\n  {patient_id}")
        for doc_type, files in sorted(documents.items()):
            print(f"    {doc_type:<18} {', '.join(files)}")
    print(f"\n  {len(catalogue)} patient(s).")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .common.doc_loader import scan_incoming
    from .pipeline import default_mode, run_pipeline

    patients = (
        sorted(scan_incoming().keys()) if args.all
        else [p.strip().upper() for p in args.patients]
    )
    if not patients:
        print("Give a patient id (e.g. P1019) or --all.")
        return 2

    mode = args.mode or default_mode()
    _print_header(f"Running the pipeline for {len(patients)} patient(s) in {mode} mode")
    print(f"  {provider_banner()}\n")

    failures = 0
    for patient_id in patients:
        print(f"\n--- {patient_id} " + "-" * (72 - len(patient_id)))

        def progress(stage: str, message: str) -> None:
            print(f"  [{stage:<10}] {message}")

        result = asyncio.run(
            run_pipeline(
                patient_id,
                mode=mode,
                generate_summary=args.summary,
                force_summary=args.force,
                use_llm_extraction=not args.no_llm,
                progress=progress,
            )
        )

        if result.report is None:
            failures += 1
            print(f"  FAILED: {'; '.join(result.errors)}")
            continue

        risk = result.report.risk
        verdict = (
            "BLOCKED" if risk.discharge_blocked
            else "HITL REQUIRED" if risk.hitl_required
            else "AUTO-APPROVE"
        )
        print(
            f"\n  → {verdict}: risk {risk.level.value} (score {risk.score}), "
            f"recommendation {risk.recommendation.value}"
        )
        print(f"    completeness {result.report.completeness.score}% · "
              f"{len(result.report.findings)} finding(s) · "
              f"elicitation {result.report.elicitation.outcome.value}")
        for finding in result.report.findings:
            print(f"      [{finding.severity.value:8}] {finding.rule_id}: "
                  f"{finding.message[:88]}")
        print(f"    summary generated: {result.summary is not None}")
        print(f"    artifacts: {', '.join(sorted(result.artifacts)) or 'none'}")
        if result.trace_url:
            print(f"    trace: {result.trace_url}")
        if result.errors:
            print(f"    degraded: {'; '.join(result.errors)}")

    print()
    return 1 if failures else 0


def cmd_ask(args: argparse.Namespace) -> int:
    from .agents import rag_agent

    _print_header("RAG Q&A")
    print(f"  Q: {args.question}\n")

    answer = asyncio.run(
        rag_agent.ask(args.question, patient_id=args.patient, top_k=args.top_k)
    )
    print(f"  A: {answer.answer}\n")
    triad = answer.triad
    print(
        f"  RAG Triad — faithfulness {triad.faithfulness:.2f} · "
        f"answer relevance {triad.answer_relevance:.2f} · "
        f"context relevance {triad.context_relevance:.2f} "
        f"({'passed' if triad.passed else 'BELOW THRESHOLD'})"
    )
    print(f"  out of context: {answer.out_of_context}")
    print("  sources:")
    for chunk in answer.chunks:
        print(f"    - {chunk.source} [{chunk.doc_type}] "
              f"similarity {chunk.score:.3f}"
              + (f" rerank {chunk.rerank_score:.3f}" if chunk.rerank_score else ""))
    triggered = [event for event in answer.guardrail_events if event.triggered]
    if triggered:
        print("  guardrails triggered:")
        for event in triggered:
            print(f"    - {event.guardrail}: {event.action}")
    return 0


def cmd_index(_args: argparse.Namespace) -> int:
    from .rag import indexing

    _print_header("Rebuilding the FAISS index")
    indexing.reset_store()
    stats = indexing.build_index()
    print(json.dumps(stats, indent=2, default=str))
    return 0


def cmd_mcp(_args: argparse.Namespace) -> int:
    from .mcp_client import ClinicalMCPClient

    async def _inspect() -> int:
        async with ClinicalMCPClient(agent_name="cli") as client:
            _print_header("MCP servers")
            print(f"  connected: {client.connected_servers or 'NONE'}")
            if not client.is_connected:
                print(
                    "\n  Start them with:\n"
                    "    python run_services.py --core"
                )
                return 1

            _print_header("Tools")
            for server, tools in (await client.list_tools()).items():
                print(f"\n  [{server}]")
                for tool in tools:
                    print(f"    {tool.get('name')}")
                    description = (tool.get("description") or "").strip()
                    if description:
                        print(f"      {description[:150]}")

            _print_header("Resources")
            for resource in await client.list_resources():
                marker = "T" if resource["templated"] else " "
                print(f"  {marker} {resource['uri']:<48} {resource['mime_type']}")

            _print_header("Prompts")
            for prompt in await client.list_prompts():
                print(f"    {prompt['name']:<38} args={prompt['arguments']}")
            return 0

    return asyncio.run(_inspect())


def cmd_agents(_args: argparse.Namespace) -> int:
    from .a2a_layer import A2AAgentClient

    _print_header("A2A AgentCard discovery (/.well-known/agent.json)")
    summary = asyncio.run(A2AAgentClient().discover_all())

    for key, info in summary.items():
        marker = "🟢" if info["online"] else "⚪"
        print(f"\n  {marker} {info['name']}")
        print(f"     key        {key}")
        print(f"     url        {info['url']}")
        print(f"     framework  {info['framework']}")
        print(f"     streaming  configured={info['configured_streaming']} "
              f"card={info['card_streaming']}")
        print(f"     push       {info['card_push_notifications']}")
        print(f"     skills     {', '.join(info['skills']) or '—'}")

    offline = [info["name"] for info in summary.values() if not info["online"]]
    if offline:
        print(f"\n  Offline: {', '.join(offline)}")
        print("  Start them with: python run_services.py --agents")
    return 0


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    configure_logging("cli")

    parser = argparse.ArgumentParser(
        prog="discharge_ai", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor", help="check the environment").set_defaults(func=cmd_doctor)
    subparsers.add_parser("scan", help="list the incoming documents").set_defaults(func=cmd_scan)

    run_parser = subparsers.add_parser("run", help="run the discharge pipeline")
    run_parser.add_argument("patients", nargs="*", help="patient ids, e.g. P1019")
    run_parser.add_argument("--all", action="store_true", help="process every patient")
    run_parser.add_argument("--mode", choices=["local", "a2a"],
                            help="execution mode (default: auto-detect)")
    run_parser.add_argument("--summary", action="store_true",
                            help="also generate the discharge summary after validation "
                                 "(off by default per PDF §2.5 — the summary is a "
                                 "reviewer-requested step)")
    run_parser.add_argument("--force", action="store_true",
                            help="with --summary, generate even if the discharge is blocked")
    run_parser.add_argument("--no-llm", action="store_true",
                            help="deterministic extraction only")
    run_parser.set_defaults(func=cmd_run)

    ask_parser = subparsers.add_parser("ask", help="ask the RAG agent a question")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--patient", help="restrict retrieval to one patient")
    ask_parser.add_argument("--top-k", type=int, default=None)
    ask_parser.set_defaults(func=cmd_ask)

    subparsers.add_parser("index", help="rebuild the FAISS index").set_defaults(func=cmd_index)
    subparsers.add_parser("mcp", help="list MCP tools/resources/prompts").set_defaults(func=cmd_mcp)
    subparsers.add_parser("agents", help="discover A2A AgentCards").set_defaults(func=cmd_agents)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    finally:
        tracing.flush()


if __name__ == "__main__":
    raise SystemExit(main())
