class ModuleManager:
    def __init__(self):
        self.modules = {}

    def register(self, name, module):
        if name in self.modules:
            raise Exception(f'{name} is already registered')

        self.modules[name] = module

    def __getattr__(self, item):
        if item in self.modules:
            return self.modules[item]
        else:
            raise Exception(f'{item} is not a valid module')
