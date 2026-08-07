#!/usr/bin/env python
"""
diagnose_heatmap.py
===================

Print exactly what the risk-heatmap panel on **2 · Validation Report** is being
handed, so a blank panel can be attributed to a data problem or a rendering one
instead of guessed at.

    python diagnose_heatmap.py            # every patient with a saved report
    python diagnose_heatmap.py P1022      # one patient
    python diagnose_heatmap.py --live P1022   # call the Analytics MCP directly

Without --live this reads the saved `Data/reports/<patient>_audit.json`, which
is the same object the dashboard renders from.  With --live it calls the
Analytics MCP server on :8201, which tells you whether the server itself is
reachable and what it returns.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from discharge_ai.settings import settings  # noqa: E402


def _reports() -> Path:
    return settings.path("reports_dir")


def inspect_saved(patient_id: str) -> None:
    path = _reports() / f"{patient_id}_audit.json"
    print(f"\n=== {patient_id} — saved report ===")
    print(f"  file: {path}")
    if not path.exists():
        print("  MISSING — this patient has not been processed yet.")
        return

    report = json.loads(path.read_text(encoding="utf-8"))
    analytics = report.get("analytics") or {}

    print(f"  report keys      : {', '.join(sorted(report))[:120]}")
    print(f"  analytics present: {bool(analytics)}")
    if not analytics:
        print("  -> analytics is EMPTY. The validator could not reach the")
        print("     Analytics MCP server (:8201) during this run, so the panel")
        print("     has nothing to draw. Re-run the pipeline with it started.")
        return

    if analytics.get("error"):
        print(f"  -> analytics ERROR: {analytics['error']}")
        return

    print(f"  analytics keys   : {', '.join(sorted(analytics))}")

    heatmap = analytics.get("heatmap") or {}
    print(f"  heatmap type     : {type(heatmap).__name__}")
    if not isinstance(heatmap, dict):
        print(f"  -> heatmap is NOT a dict; raw value: {str(heatmap)[:300]}")
        return

    print(f"  heatmap keys     : {', '.join(sorted(heatmap)) or 'NONE'}")
    cells = heatmap.get("cells")
    print(f"  cells            : {len(cells) if isinstance(cells, list) else 'MISSING'}")

    if isinstance(cells, list) and cells:
        print(f"  total_score      : {heatmap.get('total_score')}")
        print(f"  worst_domain     : {heatmap.get('worst_domain')}")
        print("  rows the panel will draw:")
        for cell in cells:
            print(
                f"    {str(cell.get('label') or cell.get('domain')):26}"
                f" score={cell.get('score'):<4}"
                f" intensity={cell.get('intensity')}"
                f" severity={cell.get('severity')}"
            )
        print("  -> DATA IS FINE. A blank panel here is a rendering problem.")
    else:
        print("  -> cells missing/empty. The panel falls back to the tool's")
        print("     markdown table if present, else an explanatory message.")


async def inspect_live(patient_id: str) -> None:
    from discharge_ai.mcp_client import ClinicalMCPClient

    print(f"\n=== {patient_id} — live Analytics MCP call ===")
    client = ClinicalMCPClient(servers=("analytics",), agent_name="diagnose")
    async with client:
        print(f"  connected servers: {client.connected_servers or 'NONE'}")
        if "analytics" not in client.connected_servers:
            print(f"  -> cannot reach {settings.mcp_analytics_url}")
            print("     start it:  python -m discharge_ai.mcp_servers.analytics_server")
            return
        payload = await client.call_tool(
            "generate_risk_heatmap",
            {"patient_id": patient_id,
             "risk_keys": ["allergy_contradiction", "followup_missing"]},
            server="analytics",
        )
        print(f"  returned type: {type(payload).__name__}")
        print(json.dumps(payload, indent=2, default=str)[:1600])


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--live"]
    live = "--live" in sys.argv

    if args:
        patients = [a.strip().upper() for a in args]
    else:
        patients = sorted(
            p.name.replace("_audit.json", "")
            for p in _reports().glob("*_audit.json")
        )
        if not patients:
            print(f"No saved reports under {_reports()} — process a patient first.")
            return 1

    for patient_id in patients:
        if live:
            asyncio.run(inspect_live(patient_id))
        else:
            inspect_saved(patient_id)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
