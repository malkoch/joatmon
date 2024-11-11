from collections import OrderedDict


class Module:
    def __init__(self, system):
        self.system = system


class ModuleManager:
    def __init__(self):
        self.modules = OrderedDict({})

    def __reversed__(self):
        for k, v in reversed(self.modules.items()):
            yield v

    def __iter__(self):
        for k, v in self.modules.items():
            yield v

    def register(self, name, module: Module):
        if name in self.modules:
            raise Exception(f'{name} is already registered')

        self.modules[name] = module

    def __getattr__(self, item):
        if item in self.modules:
            return self.modules[item]
        else:
            raise Exception(f'{item} is not a valid module')
