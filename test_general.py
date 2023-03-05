import json
import os
import importlib.util
import sys

content = json.loads(open('iva.json').read())

for scripts in content.get('scripts', []):
    if os.path.isabs(scripts) and os.path.exists(scripts):
        print(f'this is a path: {scripts}')

        spec = importlib.util.spec_from_file_location("arxiv", os.path.join(scripts, f'{"arxiv"}.py'))
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        foo.Task()
