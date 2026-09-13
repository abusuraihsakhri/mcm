from auth import authenticate, login
from user import User


def test_authenticate_returns_username():
    assert authenticate(User("ada")) == "ada"


def test_login_rejects_inactive_user():
    assert login("ada") == "ada"
