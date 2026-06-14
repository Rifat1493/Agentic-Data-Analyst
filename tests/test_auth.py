"""Standalone tests for authentication (Auth0 JWT) and authorization.

Runs FULLY OFFLINE — no Auth0, no network, no LLM, no API key:
  * Authentication is tested with a locally generated RSA keypair (we sign a
    JWT ourselves and verify it with the matching public key via the
    `signing_key` test hook).
  * Authorization is tested directly against the policy.
  * The in-graph guards are tested by calling the worker nodes directly with a
    denied principal — the guard short-circuits BEFORE any LLM/network call.

Run:  python tests/test_auth.py
"""
import sys
import os
import time
import datetime as dt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

from src.middleware.authentication import (
    authenticate, verify_token, anonymous_principal, Principal, AuthenticationError,
    DEFAULT_ROLES_CLAIM,
)
from src.middleware.authorization import check_access, permitted_agents

# --------------------------------------------------------------------------- #
# Test fixtures: a local RSA keypair + Auth0-like config                       #
# --------------------------------------------------------------------------- #
_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_PUBLIC_KEY = _PRIVATE_KEY.public_key()
_PRIVATE_PEM = _PRIVATE_KEY.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
)

ISSUER = "https://test-tenant.auth0.com/"
AUDIENCE = "https://api.agentic-data-analyst/"

# Point the verifier at our test issuer/audience/claim by overriding config
# (modules read config.<NAME> at call time, so these take effect).
from src import config
config.AUTH0_ISSUER = ISSUER
config.AUTH0_API_AUDIENCE = AUDIENCE
config.AUTH0_ROLES_CLAIM = DEFAULT_ROLES_CLAIM


def _make_token(sub: str, roles: list, *, expires_in: int = 3600,
                audience: str = AUDIENCE, issuer: str = ISSUER) -> str:
    now = int(time.time())
    claims = {
        "sub": sub,
        "iss": issuer,
        "aud": audience,
        "iat": now,
        "exp": now + expires_in,
        DEFAULT_ROLES_CLAIM: roles,
    }
    return jwt.encode(claims, _PRIVATE_PEM, algorithm="RS256", headers={"kid": "test"})


def check(label: str, condition: bool) -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}")
    assert condition, label


# --------------------------------------------------------------------------- #
# 1. Authentication                                                            #
# --------------------------------------------------------------------------- #
def test_authentication():
    print("\n=== Authentication (Auth0 JWT verification) ===")

    # Valid senior token.
    token = _make_token("auth0|senior1", ["senior"])
    p = verify_token(token, signing_key=_PUBLIC_KEY)
    check("valid token is authenticated", p.authenticated)
    check("user_id extracted from 'sub'", p.user_id == "auth0|senior1")
    check("'senior' role extracted", p.has_role("senior"))

    # No token -> anonymous.
    anon = authenticate(None)
    check("no token -> not authenticated", anon.authenticated is False)
    check("no token -> no roles", anon.roles == [])

    # Expired token -> AuthenticationError.
    expired = _make_token("auth0|x", ["senior"], expires_in=-10)
    try:
        verify_token(expired, signing_key=_PUBLIC_KEY)
        check("expired token rejected", False)
    except AuthenticationError:
        check("expired token rejected", True)

    # Wrong audience -> AuthenticationError.
    bad_aud = _make_token("auth0|x", ["senior"], audience="https://other/")
    try:
        verify_token(bad_aud, signing_key=_PUBLIC_KEY)
        check("wrong-audience token rejected", False)
    except AuthenticationError:
        check("wrong-audience token rejected", True)

    # Tampered signature (verify against a DIFFERENT key) -> rejected.
    other_pub = rsa.generate_private_key(public_exponent=65537, key_size=2048).public_key()
    try:
        verify_token(token, signing_key=other_pub)
        check("bad-signature token rejected", False)
    except AuthenticationError:
        check("bad-signature token rejected", True)


# --------------------------------------------------------------------------- #
# 2. Authorization matrix                                                      #
# --------------------------------------------------------------------------- #
def test_authorization():
    print("\n=== Authorization matrix ===")
    senior = Principal(user_id="u-senior", roles=["senior"], authenticated=True)
    regular = Principal(user_id="u-reg", roles=["analyst"], authenticated=True)
    anon = anonymous_principal()

    # data_collector: senior only.
    check("senior CAN use data_collector", check_access(senior, "data_collector")[0])
    check("regular CANNOT use data_collector", not check_access(regular, "data_collector")[0])
    check("anonymous CANNOT use data_collector", not check_access(anon, "data_collector")[0])

    # code_generator: any authenticated user.
    check("senior CAN use code_generator", check_access(senior, "code_generator")[0])
    check("regular CAN use code_generator", check_access(regular, "code_generator")[0])
    check("anonymous CANNOT use code_generator", not check_access(anon, "code_generator")[0])

    # report_generator: everyone.
    check("senior CAN use report_generator", check_access(senior, "report_generator")[0])
    check("regular CAN use report_generator", check_access(regular, "report_generator")[0])
    check("anonymous CAN use report_generator", check_access(anon, "report_generator")[0])

    # permitted_agents view.
    check("senior permitted = all 3", set(permitted_agents(senior)) ==
          {"data_collector", "code_generator", "report_generator"})
    check("regular permitted = code+report", set(permitted_agents(regular)) ==
          {"code_generator", "report_generator"})
    check("anonymous permitted = report only", permitted_agents(anon) == ["report_generator"])

    # Fail-closed on unknown agent.
    check("unknown agent denied", not check_access(senior, "rogue_agent")[0])


# --------------------------------------------------------------------------- #
# 3. In-graph guard (deterministic, no LLM/network)                           #
# --------------------------------------------------------------------------- #
def test_in_graph_guard():
    print("\n=== In-graph node guards (no LLM called when denied) ===")
    # Imported here so module import does not require network/keys earlier.
    from src.workflows.analyst_workflow import (
        data_collector_node, code_generator_node, authenticate_node,
    )
    from langchain_core.messages import HumanMessage

    regular = Principal(user_id="u-reg", roles=["analyst"], authenticated=True)
    anon = anonymous_principal()

    # A regular user hitting data_collector is blocked WITHOUT any agent call.
    state = {"messages": [HumanMessage("download AAPL")], "principal": regular.to_dict()}
    out = data_collector_node(state)
    check("regular blocked from data_collector", "data_collector" in out.get("denied", []))
    check("no data produced on denial", out.get("collected_data") is None)

    # Anonymous user blocked from code_generator.
    state = {"messages": [HumanMessage("chart it")], "principal": anon.to_dict()}
    out = code_generator_node(state)
    check("anonymous blocked from code_generator", "code_generator" in out.get("denied", []))

    # authenticate_node with no token -> anonymous principal in state.
    out = authenticate_node({"messages": []}, {"configurable": {}})
    check("authenticate_node yields anonymous when no jwt",
          out["principal"]["authenticated"] is False)


if __name__ == "__main__":
    test_authentication()
    test_authorization()
    test_in_graph_guard()
    print("\nAll auth tests passed.")
