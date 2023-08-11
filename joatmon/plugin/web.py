from joatmon import context
from joatmon.plugin.core import Plugin


class UserPlugin(Plugin):
    def __init__(self, key):
        self.key = key

    def set(self, value):
        context.set_value(self.key, value)

    def get(self):
        return context.get_value(self.key)


class TokenPlugin(Plugin):
    def __init__(self, key):
        self.key = key

    def set(self, value):
        context.set_value(self.key, value)

    def get(self):
        return context.get_value(self.key)


class IssuerPlugin(Plugin):
    def __init__(self, key):
        self.key = key

    def set(self, value):
        context.set_value(self.key, value)

    def get(self):
        return context.get_value(self.key)


class LanguagePlugin(Plugin):
    def __init__(self, key):
        self.key = key

    def set(self, value):
        context.set_value(self.key, value)

    def get(self):
        return context.get_value(self.key)


class IPPlugin(Plugin):
    def __init__(self, key):
        self.key = key

    def set(self, value):
        context.set_value(self.key, value)

    def get(self):
        return context.get_value(self.key)


class InCasePlugin(Plugin):
    def __init__(self, key):
        self.key = key

    def set(self, value):
        context.set_value(self.key, value)

    def get(self):
        return context.get_value(self.key)


class OutCasePlugin(Plugin):
    def __init__(self, key):
        self.key = key

    def set(self, value):
        context.set_value(self.key, value)

    def get(self):
        return context.get_value(self.key)


class JsonPlugin(Plugin):
    def __init__(self, key):
        self.key = key

    def set(self, value):
        context.set_value(self.key, value)

    def get(self):
        return context.get_value(self.key)


class ArgsPlugin(Plugin):
    def __init__(self, key):
        self.key = key

    def set(self, value):
        context.set_value(self.key, value)

    def get(self):
        return context.get_value(self.key)


class FormPlugin(Plugin):
    def __init__(self, key):
        self.key = key

    def set(self, value):
        context.set_value(self.key, value)

    def get(self):
        return context.get_value(self.key)


class HeadersPlugin(Plugin):
    def __init__(self, key):
        self.key = key

    def set(self, value):
        context.set_value(self.key, value)

    def get(self):
        return context.get_value(self.key)


class CookiesPlugin(Plugin):
    def __init__(self, key):
        self.key = key

    def set(self, value):
        context.set_value(self.key, value)

    def get(self):
        return context.get_value(self.key)
