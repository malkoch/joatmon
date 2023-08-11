# joatmon (jack of all trades, master of none)

[![Documentation Status](https://readthedocs.org/projects/joatmon/badge/?version=latest)](https://joatmon.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/joatmon.svg)](https://badge.fury.io/py/joatmon)
[![codecov](https://codecov.io/gh/malkoch/joatmon/branch/master/graph/badge.svg?token=LLMWHT1CN1)](https://codecov.io/gh/malkoch/joatmon)
[![GitHub license](https://img.shields.io/github/license/malkoch/joatmon)](https://github.com/malkoch/joatmon/blob/master/LICENSE)
[![Pylint](https://github.com/malkoch/joatmon/actions/workflows/pylint.yml/badge.svg)](https://github.com/malkoch/joatmon/actions/workflows/pylint.yml)
[![Python package](https://github.com/malkoch/joatmon/actions/workflows/package.yml/badge.svg)](https://github.com/malkoch/joatmon/actions/workflows/package.yml)
[![Upload Python Package](https://github.com/malkoch/joatmon/actions/workflows/release.yml/badge.svg)](https://github.com/malkoch/joatmon/actions/workflows/release.yml)

You can find more information in the [doc](https://joatmon.readthedocs.io/en/latest/).

## Installation

- Install joatmon from Pypi (recommended):

```
pip install joatmon
```

Install from Github source:

```
git clone https://github.com/malkoch/joatmon.git
cd joatmon
python setup.py install
```

## Examples

If you want to run the examples, you'll also have to install:

- **gym** by OpenAI: [Installation instruction](https://github.com/openai/gym#installation)

Once you have installed everything, you can try out a simple example:

```bash
python examples/sokoban_dqn.py
python examples/sokoban_ddpg.py
```

## How to run the tests

To run the tests locally, you'll first have to install the following dependencies:

```bash
pip install pytest pytest-xdist pep8 pytest-pep8 pytest-cov python-coveralls
```

You can then run all tests using this command:

```bash
py.test tests/.
```

If you want to check if the files conform to the PEP8 style guidelines, run the following command:

```bash
py.test --pep8
```

If you want to check the code coverage, run the following command:

```bash
py.test --cov=joatmon tests/
```
