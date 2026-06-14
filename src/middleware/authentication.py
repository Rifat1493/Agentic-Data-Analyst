"""Authentication middleware — verify Auth0-issued JWTs and build a Principal.

This runs INSIDE the LangGraph workflow (no web framework involved). The caller
passes the raw Auth0 access token into the graph via
`config={"configurable": {"jwt": "<token>"}}`; the `authenticate_node` in the
workflow calls `authenticate()` here and stores the resulting Principal in state.

Auth0 specifics
---------------
Auth0 access tokens are RS256-signed JWTs. We verify them against Auth0's public
JWKS (JSON Web Key Set), checking signature, issuer, audience and expiry. Roles
are read from a custom namespaced claim (added by an Auth0 Action/Rule) or from
the RBAC `permissions` claim.

Required environment (.env.dev)
-------------------------------
    AUTH0_DOMAIN        e.g. your-tenant.us.auth0.com
    AUTH0_API_AUDIENCE  the API identifier configured in Auth0
    AUTH0_ISSUER        (optional) defaults to https://$AUTH0_DOMAIN/
    AUTH0_ROLES_CLAIM   (optional) custom roles claim; default below
"""
from dataclasses import dataclass, field
from typing import List, Optional

import jwt
from jwt import PyJWKClient

from src import config

# Re-exported for convenience / tests.
DEFAULT_ROLES_CLAIM = config.DEFAULT_ROLES_CLAIM


class AuthenticationError(Exception):
    """Raised when a token is present but invalid (bad signature, expired, etc.)."""


@dataclass
class Principal:
    """The verified identity that flows through the graph.

    An *anonymous* principal (no token) has `authenticated=False` and no roles.
    """
    user_id: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    authenticated: bool = False

    def has_role(self, role: str) -> bool:
        return role in self.roles

    # The graph state is a TypedDict that may be checkpointed, so we store the
    # principal as a plain dict and rebuild it where needed.
    def to_dict(self) -> dict:
        return {"user_id": self.user_id, "roles": list(self.roles),
                "authenticated": self.authenticated}

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "Principal":
        data = data or {}
        return cls(
            user_id=data.get("user_id"),
            roles=list(data.get("roles", [])),
            authenticated=bool(data.get("authenticated", False)),
        )


def anonymous_principal() -> Principal:
    """The identity used when no token is supplied."""
    return Principal(user_id=None, roles=[], authenticated=False)


def _principal_from_claims(claims: dict) -> Principal:
    """Map verified Auth0 JWT claims to a Principal."""
    roles_claim = config.AUTH0_ROLES_CLAIM
    roles = claims.get(roles_claim) or claims.get("permissions") or []
    if isinstance(roles, str):
        roles = [roles]
    return Principal(user_id=claims.get("sub"), roles=list(roles), authenticated=True)


def verify_token(token: str, *, signing_key=None) -> Principal:
    """Verify an Auth0 JWT and return an authenticated Principal.

    Args:
        token: the raw JWT string.
        signing_key: TEST/override hook — a public key (PEM or key object) to
            verify against instead of fetching Auth0's JWKS. Production calls
            leave this None so the key is fetched from Auth0.

    Raises:
        AuthenticationError: if the token is invalid in any way.
    """
    audience = config.AUTH0_API_AUDIENCE
    domain = config.AUTH0_DOMAIN
    issuer = config.AUTH0_ISSUER

    try:
        if signing_key is None:
            if not domain:
                raise AuthenticationError("AUTH0_DOMAIN is not configured.")
            jwks_url = f"https://{domain}/.well-known/jwks.json"
            signing_key = PyJWKClient(jwks_url).get_signing_key_from_jwt(token).key

        claims = jwt.decode(
            token,
            signing_key,
            algorithms=["RS256"],
            audience=audience,
            issuer=issuer,
        )
    except AuthenticationError:
        raise
    except jwt.PyJWTError as exc:
        raise AuthenticationError(f"Invalid token: {exc}") from exc

    return _principal_from_claims(claims)


def authenticate(token: Optional[str], *, signing_key=None) -> Principal:
    """Entry point used by the graph.

    No token -> anonymous principal (allowed; some agents need no auth).
    A token  -> verified, or AuthenticationError if it is invalid.
    """
    if not token:
        return anonymous_principal()
    return verify_token(token, signing_key=signing_key)
