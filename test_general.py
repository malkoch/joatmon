import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--create', type=str)
parser.add_argument('--update', type=str)
parser.add_argument('--delete', type=str)
parser.add_argument('--value', type=str)

namespace, _ = parser.parse_known_args(['--create', 'hello world'])
print(namespace.create)
