from pathlib import Path

import pytest

from mcm.ingestion.repository import RepositoryIngestor
from mcm.storage.sqlite_store import SQLiteStore

DEMO_REPO = Path(__file__).resolve().parent.parent / "examples" / "app"

# Object IDs from the spec section 65 fixture, named once so the acceptance
# tests read as the scenario rather than as string manipulation.
JWT = "lib://jwt"
DECODE_CLAIMS = "repo://app/jwt_provider.py#function:decode_claims"
ENCODE_CLAIMS = "repo://app/jwt_provider.py#function:encode_claims"
VALIDATE_TOKEN = "repo://app/auth.py#function:validate_token"
CREATE_TOKEN = "repo://app/auth.py#function:create_token"
AUTHENTICATE = "repo://app/auth.py#function:authenticate"
LOGIN = "repo://app/auth.py#function:login"
TEST_AUTHENTICATE = "repo://app/tests/test_auth.py#test:test_authenticate_returns_username"
TEST_LOGIN = "repo://app/tests/test_auth.py#test:test_login_rejects_inactive_user"
USER_CLASS = "repo://app/user.py#class:User"
USER_IS_ACTIVE = "repo://app/user.py#method:User.is_active"
AUTH_FILE = "repo://app/auth.py#file"


@pytest.fixture(scope="session")
def store():
    """An in-memory store with examples/app ingested."""
    s = SQLiteStore()
    RepositoryIngestor(s).ingest(DEMO_REPO, name="app")
    yield s
    s.close()


@pytest.fixture(scope="session")
def report():
    s = SQLiteStore()
    r = RepositoryIngestor(s).ingest(DEMO_REPO, name="app")
    yield r
    s.close()


@pytest.fixture
def empty_store():
    s = SQLiteStore()
    yield s
    s.close()
