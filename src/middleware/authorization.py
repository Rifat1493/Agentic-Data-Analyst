"""Authorization middleware — per-agent access policy and the deterministic guard.

This is the SECURITY BOUNDARY. The supervisor LLM is also told which agents a
user may use (for good behaviour / UX), but the LLM is never trusted to enforce
access — `check_access` below is the hard, deterministic gate run inside each
protected worker node.

Policy for this project
-----------------------
    data_collector   -> authenticated AND 'senior' role
    code_generator   -> authenticated (any role)
    report_generator -> open to everyone (even anonymous)

Unknown agents default to DENY (fail closed).
"""
from typing import Tuple

from src.middleware.authentication import Principal


# Central, declarative policy — easy to read, test, and extend.
AGENT_POLICY = {
    "data_collector":   {"require_auth": True,  "required_role": "senior"},
    "code_generator":   {"require_auth": True,  "required_role": None},
    "report_generator": {"require_auth": False, "required_role": None},
}


class AuthorizationError(Exception):
    """Raised by `require_access` when a principal may not use an agent."""


def check_access(principal: Principal, agent_name: str) -> Tuple[bool, str]:
    """Return (allowed, reason). `reason` is empty when allowed.

    Fails closed: an agent not present in AGENT_POLICY is denied.
    """
    policy = AGENT_POLICY.get(agent_name)
    if policy is None:
        return False, f"Unknown agent '{agent_name}' is not permitted."

    if policy["require_auth"] and not principal.authenticated:
        return False, f"'{agent_name}' requires authentication."

    role = policy["required_role"]
    if role and not principal.has_role(role):
        return False, f"'{agent_name}' requires the '{role}' role."

    return True, ""


def require_access(principal: Principal, agent_name: str) -> None:
    """Raise AuthorizationError if `principal` may not use `agent_name`."""
    allowed, reason = check_access(principal, agent_name)
    if not allowed:
        raise AuthorizationError(reason)


def permitted_agents(principal: Principal) -> list:
    """List the agents this principal is allowed to use (for the supervisor)."""
    return [name for name in AGENT_POLICY if check_access(principal, name)[0]]
