"""Thin wrapper over the JWT library used by the auth layer."""

import jwt

SECRET = "dev-secret"


def encode_claims(claims):
    return jwt.encode(claims, SECRET, algorithm="HS256")


def decode_claims(token):
    return jwt.decode(token, SECRET, algorithms=["HS256"])
