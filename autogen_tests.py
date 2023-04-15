import importlib.util
import inspect
import os

from joatmon.utility import to_snake_string

for path, folders, files in os.walk('joatmon'):
    if '__pycache__' in path:
        continue

    for file in files:
        if '.py' not in file:
            continue

        module_path = os.path.join(path, file)

        spec = importlib.util.spec_from_file_location(path, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        classes = inspect.getmembers(module, predicate=lambda x: inspect.isclass(x))
        module_classes = filter(lambda x: x[1].__module__ == module.__name__, classes)

        module_name = module.__name__.replace('\\', '.') + '.' + file.replace('.py', '')

        module_info = {
            'name': module_name,
            'classes': {

            },
            'functions': [

            ]
        }

        for module_class in module_classes:
            class_methods = inspect.getmembers(module_class[1], predicate=lambda x: inspect.isfunction(x))
            class_methods = filter(lambda x: x[1].__module__ == module.__name__, class_methods)
            class_methods = filter(lambda x: not x[0].startswith('__'), class_methods)
            class_methods = filter(lambda x: not x[0].endswith('__'), class_methods)
            class_methods = map(lambda x: x[0], class_methods)

            module_info['classes'][module_class[1].__name__] = list(class_methods)

        methods = inspect.getmembers(module, predicate=lambda x: inspect.isfunction(x))
        module_methods = filter(lambda x: x[1].__module__ == module.__name__, methods)
        module_methods = filter(lambda x: not x[0].startswith('__'), module_methods)
        module_methods = filter(lambda x: not x[0].endswith('__'), module_methods)
        module_methods = map(lambda x: x[0], module_methods)

        module_info['functions'] = list(module_methods)

        # print(json.dumps(module_info, indent=4))

        base_folder = 'tests'
        if not os.path.exists(base_folder):
            os.mkdir(base_folder)

        module_folder = os.path.join(base_folder, *module_name.split('.')[1:-1])
        if not os.path.exists(module_folder):
            os.mkdir(module_folder)

        if file == '__init__.py':
            module_test_path = os.path.join(module_folder, f'test_{"_".join(module_name.split(".")[1:-1])}.py')
        else:
            module_test_path = os.path.join(module_folder, f'test_{"_".join(module_name.split(".")[1:])}.py')

        with open(module_test_path, 'w') as f:
            class_function_tests = '\n\n\n'.join(
                [
                    f'def test_{to_snake_string(class_name)}_{class_function}():\n\tassert True is True'
                    for class_name, class_functions in module_info['classes'].items()
                    for class_function in class_functions
                ]
            )
            function_tests = '\n\n\n'.join(
                [
                    f'def test_{function}():\n\tassert True is True'
                    for function in module_info['functions']
                ]
            )

            f.write(f'''import pytest\n\n\n{class_function_tests}\n\n\n{function_tests}\n\n\nif __name__ == '__main__':\n\tpytest.main([__file__])\n''')
