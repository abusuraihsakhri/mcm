"""User records."""


class User:
    def __init__(self, username, active=True):
        self.username = username
        self.active = active

    def is_active(self):
        return self.active


def load_user(username):
    return User(username)
