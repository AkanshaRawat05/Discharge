#!/usr/bin/env python
"""
run_services.py
===============

One-command launcher for the whole system.

    python run_services.py                # start every service
    python run_services.py --core          # EHR + both MCP servers only
    python run_services.py --list          # show the port map
    python run_services.py --status        # which ports are already listening
    python run_services.py --only ehr mcp_primary extractor
    python run_services.py --stop          # stop anything listening on our ports

Each service runs as its own child process (that is the point — the six agents
are genuinely separate A2A peers).  Logs are interleaved on this console with a
per-service prefix; Ctrl-C shuts everything down.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
VENV_PYTHON = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
PYTHON = str(VENV_PYTHON if VENV_PYTHON.exists() else sys.executable)


# --------------------------------------------------------------------------- #
#  Service catalogue  (start order matters: infrastructure before agents)
# --------------------------------------------------------------------------- #
class Service:
    def __init__(self, key: str, label: str, module: str, port: int,
                 group: str, args: list[str] | None = None) -> None:
        self.key = key
        self.label = label
        self.module = module
        self.port = port
        self.group = group
        self.args = args or []
        self.process: subprocess.Popen | None = None


SERVICES: list[Service] = [
    # ---- infrastructure ----------------------------------------------------
    Service("ehr", "Mock EHR System", "discharge_ai.ehr.server", 8050, "core"),
    Service("mcp_primary", "MCP Clinical Tools Server",
            "discharge_ai.mcp_servers.primary_server", 8200, "core"),
    Service("mcp_analytics", "MCP Analytics Server",
            "discharge_ai.mcp_servers.analytics_server", 8201, "core"),
    # ---- A2A agents --------------------------------------------------------
    Service("extractor", "Clinical Extractor Agent (LangGraph)",
            "discharge_ai.agents.extractor_agent", 8100, "agents"),
    Service("validator", "Clinical Validation Agent (LangGraph)",
            "discharge_ai.agents.validator_agent", 8101, "agents"),
    Service("normalizer", "Clinical Normalizer Agent (LangGraph)",
            "discharge_ai.agents.normalizer_agent", 8102, "agents"),
    Service("monitor", "Discharge Monitor Agent (ADK)",
            "discharge_ai.agents.monitor_agent", 8103, "agents"),
    Service("summary", "Discharge Summary Generator (ADK, streaming)",
            "discharge_ai.agents.summary_agent", 8104, "agents"),
    Service("rag", "Clinical RAG Q&A Agent (Agno, streaming)",
            "discharge_ai.agents.rag_agent", 8105, "agents"),
    # ---- user interfaces ---------------------------------------------------
    Service("orchestrator", "Host Orchestrator (ADK + Gradio)",
            "discharge_ai.agents.orchestrator", 8083, "ui"),
    Service("dashboard", "Streamlit HITL Dashboard", "", 8501, "ui"),
]

SERVICE_BY_KEY = {service.key: service for service in SERVICES}


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def child_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{SRC}{os.pathsep}{existing}" if existing else str(SRC)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def port_in_use(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_for_port(port: int, timeout: float = 45.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_in_use(port):
            return True
        time.sleep(0.4)
    return False


def command_for(service: Service) -> list[str]:
    if service.key == "dashboard":
        return [
            PYTHON, "-m", "streamlit", "run", str(ROOT / "dashboard" / "app.py"),
            "--server.port", str(service.port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ]
    return [PYTHON, "-m", service.module, *service.args]


def pump_output(service: Service) -> None:
    """Relay a child's output to this console with a service prefix."""
    stream = service.process.stdout if service.process else None
    if stream is None:
        return
    prefix = f"[{service.key:<13}]"
    for raw in iter(stream.readline, b""):
        line = raw.decode("utf-8", errors="replace").rstrip()
        if line:
            print(f"{prefix} {line}", flush=True)


def start(service: Service) -> bool:
    if port_in_use(service.port):
        print(f"  ~ {service.label:<48} port {service.port} already in use — skipped")
        return True

    service.process = subprocess.Popen(
        command_for(service),
        cwd=str(ROOT),
        env=child_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    threading.Thread(target=pump_output, args=(service,), daemon=True).start()

    if wait_for_port(service.port, timeout=60.0):
        print(f"  + {service.label:<48} http://127.0.0.1:{service.port}")
        return True

    if service.process.poll() is not None:
        print(f"  ! {service.label:<48} exited immediately "
              f"(code {service.process.returncode}) — see the log above")
        return False

    print(f"  ? {service.label:<48} started but port {service.port} is not "
          "listening yet; continuing")
    return True


def stop_all(services: list[Service]) -> None:
    print("\nShutting down…")
    for service in reversed(services):
        process = service.process
        if process is None or process.poll() is not None:
            continue
        try:
            if os.name == "nt":
                process.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            else:
                process.terminate()
        except Exception:  # noqa: BLE001
            pass

    deadline = time.time() + 8
    for service in reversed(services):
        process = service.process
        if process is None:
            continue
        try:
            process.wait(timeout=max(0.5, deadline - time.time()))
        except Exception:  # noqa: BLE001
            process.kill()
        print(f"  - stopped {service.label}")


def kill_ports() -> None:
    """Free every port in the map (useful after a crash left orphans)."""
    ports = {service.port for service in SERVICES}
    if os.name != "nt":
        print("--stop is implemented for Windows; on POSIX use: lsof -ti:PORT | xargs kill")
        return

    import re

    output = subprocess.run(
        ["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True
    ).stdout
    killed: set[str] = set()
    for line in output.splitlines():
        match = re.search(r":(\d+)\s+\S+\s+LISTENING\s+(\d+)", line)
        if not match:
            continue
        port, pid = int(match.group(1)), match.group(2)
        if port in ports and pid not in killed and pid != "0":
            subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True)
            killed.add(pid)
            print(f"  - killed pid {pid} on port {port}")
    if not killed:
        print("  nothing was listening on the project ports")


def show_list() -> None:
    print(f"\n{'KEY':<14} {'PORT':<6} {'GROUP':<8} SERVICE")
    print("-" * 78)
    for service in SERVICES:
        print(f"{service.key:<14} {service.port:<6} {service.group:<8} {service.label}")


def show_status() -> None:
    print(f"\n{'KEY':<14} {'PORT':<6} {'STATUS':<12} SERVICE")
    print("-" * 78)
    for service in SERVICES:
        state = "LISTENING" if port_in_use(service.port) else "-"
        print(f"{service.key:<14} {service.port:<6} {state:<12} {service.label}")
    print()


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--core", action="store_true",
                        help="start only the EHR + both MCP servers")
    parser.add_argument("--agents", action="store_true",
                        help="start core + the six A2A agents (no UIs)")
    parser.add_argument("--only", nargs="+", metavar="KEY",
                        help="start only these service keys")
    parser.add_argument("--list", action="store_true", help="print the port map and exit")
    parser.add_argument("--status", action="store_true",
                        help="show which ports are listening and exit")
    parser.add_argument("--stop", action="store_true",
                        help="kill whatever is listening on the project ports")
    args = parser.parse_args()

    if args.list:
        show_list()
        return 0
    if args.status:
        show_status()
        return 0
    if args.stop:
        kill_ports()
        return 0

    if args.only:
        unknown = [key for key in args.only if key not in SERVICE_BY_KEY]
        if unknown:
            print(f"Unknown service key(s): {', '.join(unknown)}")
            show_list()
            return 2
        selected = [SERVICE_BY_KEY[key] for key in _ordered_keys(args.only)]
    elif args.core:
        selected = [s for s in SERVICES if s.group == "core"]
    elif args.agents:
        selected = [s for s in SERVICES if s.group in {"core", "agents"}]
    else:
        selected = list(SERVICES)

    if not VENV_PYTHON.exists():
        print(f"! No virtualenv found at {VENV_PYTHON}.")
        print("  Create it first:  python -m venv .venv && "
              ".venv\\Scripts\\pip install -r requirements.txt")
        print(f"  Falling back to {PYTHON}\n")

    if not (ROOT / ".env").exists():
        print("! No .env file found — copy .env.example to .env and add your keys.\n")

    print(f"Starting {len(selected)} service(s) with {PYTHON}\n")
    failures: list[str] = []
    try:
        for service in selected:
            if not start(service):
                failures.append(service.label)

        print("\n" + "=" * 78)
        print("  Streamlit HITL Dashboard   http://127.0.0.1:8501")
        print("  Host Orchestrator (Gradio) http://127.0.0.1:8083")
        print("  Mock EHR API docs          http://127.0.0.1:8050/docs")
        print("=" * 78)
        if failures:
            print(f"  {len(failures)} service(s) failed to start: {', '.join(failures)}")
        print("  Ctrl-C to stop everything.\n")

        while True:
            time.sleep(1)
            for service in selected:
                if service.process is not None and service.process.poll() is not None:
                    print(f"[{service.key}] exited with code {service.process.returncode}")
                    service.process = None

    except KeyboardInterrupt:
        pass
    finally:
        stop_all(selected)

    return 1 if failures else 0


def _ordered_keys(keys: list[str]) -> list[str]:
    """Keep the catalogue's start order regardless of the order given."""
    wanted = set(keys)
    return [service.key for service in SERVICES if service.key in wanted]


if __name__ == "__main__":
    raise SystemExit(main())
