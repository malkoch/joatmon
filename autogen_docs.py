from __future__ import (
    print_function,
    unicode_literals
)

import inspect
import os
import re
import shutil

from joatmon.ai.callback import (
    CallbackList,
    CoreCallback,
    Loader,
    Renderer,
    TrainLogger,
    ValidationLogger,
    Visualizer
)
from joatmon.ai.memory import (
    CoreMemory,
    RingMemory
)
from joatmon.ai.models.core import CoreModel
from joatmon.ai.models.reinforcement.hybrid.ddpg import DDPGModel
from joatmon.ai.models.reinforcement.q_learning.dqn import DQNModel
from joatmon.ai.policy import (
    CorePolicy,
    EpsilonGreedyPolicy,
    GreedyQPolicy
)
from joatmon.ai.random import (
    CoreRandom,
    GaussianRandom,
    OrnsteinUhlenbeck
)
from joatmon.game import (
    CubeEnv,
    Puzzle2048,
    SokobanEnv,
    TilesEnv
)
from joatmon.game.core import CoreEnv


# need to add the buffer as well


def run():
    pages = [
        {
            'page': 'core.md',
            #            'all_module_classes': [core]
        },

        {
            'page': 'callback/overview.md',
            'classes': [CoreCallback],
            'functions': [CoreCallback.on_action_begin, CoreCallback.on_action_end, CoreCallback.on_agent_begin, CoreCallback.on_agent_end,
                          CoreCallback.on_episode_begin, CoreCallback.on_episode_end, CoreCallback.on_replay_begin, CoreCallback.on_replay_end]
        },
        {
            'page': 'callback/callbacklist.md',
            'classes': [CallbackList],
            'functions': [CallbackList.on_action_begin, CallbackList.on_action_end, CallbackList.on_agent_begin, CallbackList.on_agent_end,
                          CallbackList.on_episode_begin, CallbackList.on_episode_end, CallbackList.on_replay_begin, CallbackList.on_replay_end]
        },
        {
            'page': 'callback/renderer.md',
            'classes': [Renderer],
            'functions': [Renderer.on_action_end, Renderer.on_episode_begin]
        },
        {
            'page': 'callback/trainlog.md',
            'classes': [TrainLogger],
            'functions': [TrainLogger.on_agent_begin, TrainLogger.on_episode_begin, TrainLogger.on_episode_end, TrainLogger.on_replay_end]
        },
        {
            'page': 'callback/validationlog.md',
            'classes': [ValidationLogger],
            'functions': [ValidationLogger.on_agent_begin, ValidationLogger.on_episode_begin, ValidationLogger.on_episode_end]
        },
        {
            'page': 'callback/visualizer.md',
            'classes': [Visualizer],
            'functions': [Visualizer.on_action_begin]
        },
        {
            'page': 'callback/wloader.md',
            'classes': [Loader],
            'functions': [Loader.on_agent_begin, Loader.on_agent_end, Loader.on_episode_end]
        },

        {
            'page': 'game/overview.md',
            'classes': [CoreEnv],
            'functions': [CoreEnv.close, CoreEnv.reset, CoreEnv.step, CoreEnv.seed, CoreEnv.render]
        },
        {
            'page': 'game/puzzle.md',
            'classes': [Puzzle2048],
            'functions': [Puzzle2048.close, Puzzle2048.reset, Puzzle2048.step, Puzzle2048.seed, Puzzle2048.render]
        },
        {
            'page': 'game/rcube.md',
            'classes': [CubeEnv],
            'functions': [CubeEnv.close, CubeEnv.reset, CubeEnv.step, CubeEnv.seed, CubeEnv.render]
        },
        {
            'page': 'game/sokoban.md',
            'classes': [SokobanEnv],
            'functions': [SokobanEnv.close, SokobanEnv.reset, SokobanEnv.step, SokobanEnv.seed, SokobanEnv.render]
        },
        {
            'page': 'game/tiles.md',
            'classes': [TilesEnv],
            'functions': [TilesEnv.close, TilesEnv.reset, TilesEnv.step, TilesEnv.seed, TilesEnv.render]
        },

        {
            'page': 'memory/overview.md',
            'classes': [CoreMemory],
            'functions': [CoreMemory.remember, CoreMemory.sample]
        },
        {
            'page': 'memory/ring.md',
            'classes': [RingMemory],
            'functions': [RingMemory.remember, RingMemory.sample]
        },

        {
            'page': 'models/overview.md',
            'classes': [CoreModel],
            'functions': [CoreModel.load, CoreModel.save, CoreModel.predict, CoreModel.train, CoreModel.evaluate]
        },
        {
            'page': 'models/ddpg.md',
            'classes': [DDPGModel],
            'functions': [DDPGModel.load, DDPGModel.save, DDPGModel.predict, DDPGModel.train, DDPGModel.evaluate]
        },
        {
            'page': 'models/dqn.md',
            'classes': [DQNModel],
            'functions': [DQNModel.load, DQNModel.save, DQNModel.predict, DQNModel.train, DQNModel.evaluate]
        },

        {
            'page': 'policy/overview.md',
            'classes': [CorePolicy],
            'functions': [CorePolicy.reset, CorePolicy.decay, CorePolicy.use_network]
        },
        {
            'page': 'policy/epsgreedy.md',
            'classes': [EpsilonGreedyPolicy],
            'functions': [EpsilonGreedyPolicy.reset, EpsilonGreedyPolicy.decay, EpsilonGreedyPolicy.use_network]
        },
        {
            'page': 'policy/greedy.md',
            'classes': [GreedyQPolicy],
            'functions': [GreedyQPolicy.reset, GreedyQPolicy.decay, GreedyQPolicy.use_network]
        },

        {
            'page': 'random/overview.md',
            'classes': [CoreRandom],
            'functions': [CoreRandom.reset, CoreRandom.decay, CoreRandom.sample]
        },
        {
            'page': 'random/gauss.md',
            'classes': [GaussianRandom],
            'functions': [GaussianRandom.reset, GaussianRandom.decay, GaussianRandom.sample]
        },
        {
            'page': 'random/ou.md',
            'classes': [OrnsteinUhlenbeck],
            'functions': [OrnsteinUhlenbeck.reset, OrnsteinUhlenbeck.decay, OrnsteinUhlenbeck.sample]
        }
    ]

    hzai_prefix = 'joatmon.'

    def get_classes_ancestors(_classes):
        ancestors = []
        for _cls in _classes:
            ancestors += _cls.__bases__
        filtered_ancestors = []
        for ancestor in ancestors:
            if ancestor.__name__ in ['abstract']:
                continue
            filtered_ancestors.append(ancestor)
        if filtered_ancestors:
            return filtered_ancestors + get_classes_ancestors(filtered_ancestors)
        else:
            return filtered_ancestors

    def get_class_that_defined_method(meth):
        if inspect.ismethod(meth):
            for _cls in inspect.getmro(meth.__self__.__class__):
                if _cls.__dict__.get(meth.__name__) is meth:
                    return _cls
            meth = meth.__func__  # fallback to __qualname__ parsing
        if inspect.isfunction(meth):
            _cls = getattr(inspect.getmodule(meth), meth.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0])
            if isinstance(_cls, type):
                return _cls
        return getattr(meth, '__objclass__', None)  # handle special descriptor objects

    def get_function_signature(_func, method=True):
        class_attr = get_class_that_defined_method(_func)
        class_module = class_attr.__module__
        class_name = class_attr.__name__

        sign = getattr(_func, '_legacy_support_signature', None)
        if sign is None:
            sign = inspect.getfullargspec(_func)
        defaults = sign.defaults
        if method:
            args = sign.args[1:]
        else:
            args = sign.args
        if defaults:
            kwargs = zip(args[-len(defaults):], defaults)
            args = args[:-len(defaults)]
        else:
            kwargs = []
        st = '%s.%s(' % (_func.__module__, _func.__name__)
        for a in args:
            st += str(a) + ', '
        for a, v in kwargs:
            if isinstance(v, str):
                v = '\'' + v + '\''
            st += str(a) + '=' + str(v) + ', '
        if kwargs or args:
            st = st[:-2] + ')'
        else:
            st += ')'

        return class_module + '.' + class_name + '.' + st

    def get_class_signature(_cls):
        try:
            class_signature = get_function_signature(_cls.__init__)
            class_signature = class_signature.replace('__init__', _cls.__name__)
        except Exception as ex:
            print(str(ex))
            # in case the class inherits from abstract and does not
            # define __init__
            class_signature = _cls.__module__ + '.' + _cls.__name__ + '()'
        return class_signature

    def class_to_source_link(_cls):
        module_name = _cls.__module__
        assert module_name.startswith(hzai_prefix)
        _path = module_name.replace('.', '/')
        _path += '.py'
        line = inspect.getsourcelines(_cls)[-1]
        link = 'https://github.com/malkoch/joatmon/blob/master/' + _path + '#L' + str(line)
        return '[[source]](' + link + ')'

    def function_to_source_link(fn):
        module_name = fn.__module__
        assert module_name.startswith(hzai_prefix)
        _path = module_name.replace('.', '/')
        _path += '.py'
        line = inspect.getsourcelines(fn)[-1]
        link = 'https://github.com/malkoch/joatmon/blob/master/' + _path + '#L' + str(line)
        return '[[source]](' + link + ')'

    def code_snippet(snippet):
        result = '```python\n'
        result += snippet + '\n'
        result += '```\n'
        return result

    def process_class_docstring(_docstring):
        _docstring = re.sub(r'\n    # (.*)\n', r'\n    __\1__\n\n', _docstring)
        _docstring = re.sub(r'    ([^\s\\]+) \((.*)\n', r'    - __\1__ (\2\n', _docstring)
        _docstring = _docstring.replace('    ' * 5, '\t\t')
        _docstring = _docstring.replace('    ' * 3, '\t')
        _docstring = _docstring.replace('    ', '')
        return _docstring

    def process_function_docstring(_docstring):
        _docstring = re.sub(r'\n    # (.*)\n', r'\n    __\1__\n\n', _docstring)
        _docstring = re.sub(r'\n        # (.*)\n', r'\n        __\1__\n\n', _docstring)
        _docstring = re.sub(r'    ([^\s\\]+) \((.*)\n', r'    - __\1__ (\2\n', _docstring)
        _docstring = _docstring.replace('    ' * 6, '\t\t')
        _docstring = _docstring.replace('    ' * 4, '\t')
        _docstring = _docstring.replace('    ', '')
        return _docstring

    print('Cleaning up existing sources directory.')
    if os.path.exists('docs/source'):
        shutil.rmtree('docs/source')

    print('Populating sources directory with templates.')
    for subdir, dirs, fnames in os.walk('docs/templates'):
        for fname in fnames:
            new_subdir = subdir.replace('templates', 'source')
            if not os.path.exists(new_subdir):
                os.makedirs(new_subdir)
            if fname[-3:] == '.md':
                fpath = os.path.join(subdir, fname)
                new_fpath = fpath.replace('templates', 'source')
                shutil.copy(fpath, new_fpath)

    # Take care of index page.
    readme = open('README.md').read()
    index = open('docs/templates/index.md').read()
    index = index.replace('{{autogenerated}}', readme[readme.find('##'):])
    f = open('docs/source/index.md', 'w')
    f.write(index)
    f.close()

    print('Starting autogeneration.')
    for page_data in pages:
        blocks = []

        functions = page_data.get('functions', [])
        for module in page_data.get('all_module_functions', []):
            module_functions = []
            for name in dir(module):
                if name[0] == '_':
                    continue
                module_member = getattr(module, name)
                if inspect.isfunction(module_member):
                    func = module_member
                    if module.__name__ in func.__module__:
                        if func not in module_functions:
                            module_functions.append(func)
            module_functions.sort(key=lambda x: id(x))
            functions += module_functions

        classes = page_data.get('classes', [])
        for module in page_data.get('all_module_classes', []):
            module_classes = []
            for name in dir(module):
                if name[0] == '_':
                    continue
                module_member = getattr(module, name)
                if inspect.isclass(module_member):
                    cls = module_member
                    if cls.__module__ == module.__name__:
                        if cls not in module_classes:
                            module_classes.append(cls)
            # module_classes.sort(key=lambda x: id(x))  # change this to str(x)
            classes += module_classes

        for cls in classes:
            subblocks = []
            signature = get_class_signature(cls)
            subblocks.append('<span style="float:right;">' + class_to_source_link(cls) + '</span>')
            subblocks.append('### ' + cls.__name__ + '\n')
            subblocks.append(code_snippet(signature))
            docstring = cls.__doc__
            if docstring:
                subblocks.append(process_class_docstring(docstring))
            blocks.append('\n'.join(subblocks))

        for func in functions:
            subblocks = []
            signature = get_function_signature(func, method=False)
            signature = signature.replace(func.__module__ + '.', '')
            subblocks.append('<span style="float:right;">' + function_to_source_link(func) + '</span>')
            subblocks.append('### ' + func.__name__ + '\n')
            subblocks.append(code_snippet(signature))
            docstring = func.__doc__
            if docstring:
                subblocks.append(process_function_docstring(docstring))
            blocks.append('\n\n'.join(subblocks))

        if not blocks:
            raise RuntimeError('Found no content for page ' + page_data['page'])

        mkdown = '\n----\n\n'.join(blocks)
        # save module page.
        # Either insert content into existing page,
        # or create page otherwise
        page_name = page_data['page']
        path = os.path.join('docs/source', page_name)
        if os.path.exists(path):
            template = open(path).read()
            assert '{{autogenerated}}' in template, ('Template found for ' + path + ' but missing {{autogenerated}} tag.')
            mkdown = template.replace('{{autogenerated}}', mkdown)
            print('...inserting autogenerated content into template:', path)
        else:
            print('...creating new page with autogenerated content:', path)
        subdir = os.path.dirname(path)
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        open(path, 'w').write(mkdown)


if __name__ == '__main__':
    run()
