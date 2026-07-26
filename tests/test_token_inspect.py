"""Run:  pytest tests/test_token_inspect.py -v -s"""
import http.client, json, os
from dotenv import load_dotenv
import jwt

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env.dev"))

USERNAME = ""
PASSWORD = ""


def test_login():
    conn = http.client.HTTPSConnection(os.environ["AUTH0_DOMAIN"])
    conn.request("POST", "/oauth/token", json.dumps({
        "grant_type":    "password",
        "client_id":     os.environ["AUTH0_CLIENT_ID"],
        "client_secret": os.environ["AUTH0_CLIENT_SECRET"],
        "audience":      os.environ["AUTH0_AUDIENCE"],
        "username":      USERNAME,
        "password":      PASSWORD,
        "scope":         "openid profile email roles",
    }), {"content-type": "application/json"})

    data = json.loads(conn.getresponse().read())
    token = data.get("access_token")
    assert token, f"Login failed: {data}"

    claims = jwt.decode(token, options={"verify_signature": False})
    print(f"\n  token  : {token[:80]}...")
    print(f"  user   : {claims.get('sub')}")
    print(f"  roles  : {claims.get('permissions') or claims.get('https://agentic-data-analyst/roles') or []}")
    print(f"\n  all claims:")
    for k, v in claims.items():
        print(f"    {k}: {v}")

if __name__ == "__main__":
    test_login()