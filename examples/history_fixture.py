"""Build a small Git repository with a regression in its history.

Used by the tests and by ``mcm history-demo``. The story is four commits:

    1  initial auth: create_token, validate_token, authenticate
    2  add login and the User model
    3  add tests
    4  a claim-name change in jwt_provider that breaks authentication

Commit 4 is the regression. It edits ``decode_claims``, which nothing in auth.py
mentions by name, so finding it means following the dependency chain rather than
searching for the symptom.

Run directly to build one::

    python examples/history_fixture.py /tmp/demo-repo
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

JWT_PROVIDER_V1 = '''"""Thin wrapper over the JWT library used by the auth layer."""

import jwt

SECRET = "dev-secret"


def encode_claims(claims):
    return jwt.encode(claims, SECRET, algorithm="HS256")


def decode_claims(token):
    return jwt.decode(token, SECRET, algorithms=["HS256"])
'''

# The regression: claims are now returned under a different key, so every caller
# that reads "sub" gets None.
JWT_PROVIDER_V2 = JWT_PROVIDER_V1.replace(
    '''def decode_claims(token):
    return jwt.decode(token, SECRET, algorithms=["HS256"])
''',
    '''def decode_claims(token):
    payload = jwt.decode(token, SECRET, algorithms=["HS256"])
    return {"subject": payload.get("sub")}
''',
)

AUTH_V1 = '''"""Authentication entry points."""

from jwt_provider import decode_claims, encode_claims


def create_token(user):
    return encode_claims({"sub": user.username})


def validate_token(token):
    claims = decode_claims(token)
    return claims.get("sub")


def authenticate(user):
    token = create_token(user)
    return validate_token(token)
'''

AUTH_V2 = AUTH_V1.replace(
    '''from jwt_provider import decode_claims, encode_claims
''',
    '''from jwt_provider import decode_claims, encode_claims
from user import load_user
''',
) + '''

def login(username):
    user = load_user(username)
    if not user.is_active():
        return None
    return authenticate(user)
'''

USER = '''"""User records."""


class User:
    def __init__(self, username, active=True):
        self.username = username
        self.active = active

    def is_active(self):
        return self.active


def load_user(username):
    return User(username)
'''

TEST_AUTH = '''from auth import authenticate, login
from user import User


def test_authenticate_returns_username():
    assert authenticate(User("ada")) == "ada"


def test_login_rejects_inactive_user():
    assert login("ada") == "ada"
'''

COMMITS = [
    ("add token creation and validation",
     {"jwt_provider.py": JWT_PROVIDER_V1, "auth.py": AUTH_V1}),
    ("add login and the user model",
     {"user.py": USER, "auth.py": AUTH_V2}),
    ("add authentication tests",
     {"tests/__init__.py": "", "tests/test_auth.py": TEST_AUTH}),
    ("rename claim key returned by decode_claims",
     {"jwt_provider.py": JWT_PROVIDER_V2}),
]


def build(dest: Path | str) -> Path:
    """Create the repository at ``dest`` and return its path."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    _git(dest, "init", "-q")
    _git(dest, "config", "user.email", "fixture@example.invalid")
    _git(dest, "config", "user.name", "Fixture Author")
    _git(dest, "config", "commit.gpgsign", "false")

    for index, (subject, files) in enumerate(COMMITS):
        for relpath, content in files.items():
            path = dest / relpath
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        _git(dest, "add", "-A")
        # Distinct, increasing dates so commit ordering is unambiguous.
        stamp = f"2026-03-{index + 1:02d}T10:00:00+00:00"
        _git(dest, "-c", f"user.name=Fixture Author", "commit", "-q", "-m", subject,
             "--date", stamp, env_date=stamp)
    return dest


def _git(cwd: Path, *args: str, env_date: str | None = None) -> None:
    import os

    env = os.environ.copy()
    if env_date:
        env["GIT_AUTHOR_DATE"] = env_date
        env["GIT_COMMITTER_DATE"] = env_date
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, env=env)


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "demo-repo")
    print(build(target))
