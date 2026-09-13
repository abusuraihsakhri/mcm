"""Authentication entry points."""

from jwt_provider import decode_claims, encode_claims
from user import load_user


def create_token(user):
    return encode_claims({"sub": user.username})


def validate_token(token):
    claims = decode_claims(token)
    return claims.get("sub")


def authenticate(user):
    token = create_token(user)
    return validate_token(token)


def login(username):
    user = load_user(username)
    if not user.is_active():
        return None
    return authenticate(user)
