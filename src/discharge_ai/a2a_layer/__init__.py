"""
Agent-to-Agent (A2A) Protocol infrastructure.

    cards.py     AgentCard construction from configs/agent_config.yaml
    auth.py      shared-secret `X-Agent-Auth-Token` enforcement + client interceptor
    executor.py  base AgentExecutor bridging a plain handler onto A2A events
    server.py    run an agent as an A2A server (streaming + non-streaming)
    client.py    A2A client: send_message() / send_message_streaming()

Named `a2a_layer` rather than `a2a` so it never shadows the installed `a2a-sdk`
package inside this source tree.
"""

from .cards import build_agent_card, skill_for  # noqa: F401
from .client import A2AAgentClient, call_agent, stream_agent  # noqa: F401
from .executor import HandlerAgentExecutor, StreamEvent  # noqa: F401
from .server import run_agent_server  # noqa: F401
