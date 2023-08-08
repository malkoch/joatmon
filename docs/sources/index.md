# joatmon (jack of all trades, master of none)

[![Documentation Status](https://readthedocs.org/projects/joatmon/badge/?version=latest)](https://joatmon.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/joatmon.svg)](https://badge.fury.io/py/joatmon)
[![codecov](https://codecov.io/gh/malkoch/joatmon/branch/master/graph/badge.svg?token=LLMWHT1CN1)](https://codecov.io/gh/malkoch/joatmon)
[![GitHub license](https://img.shields.io/github/license/malkoch/joatmon)](https://github.com/malkoch/joatmon/blob/master/LICENSE)
[![Pylint](https://github.com/malkoch/joatmon/actions/workflows/pylint.yml/badge.svg)](https://github.com/malkoch/joatmon/actions/workflows/pylint.yml)
[![Python package](https://github.com/malkoch/joatmon/actions/workflows/python-package.yml/badge.svg)](https://github.com/malkoch/joatmon/actions/workflows/python-package.yml)
[![Upload Python Package](https://github.com/malkoch/joatmon/actions/workflows/python-publish.yml/badge.svg)](https://github.com/malkoch/joatmon/actions/workflows/python-publish.yml)

## What is included?

As of today, the following algorithms have been implemented:

As of today, the following environments have been implemented:

- [ ] Rubick's Cube
- [ ] 2048 Puzzle
- [x] Sokoban
- [ ] Game of 15
- [ ] Chess

As of today, the following networks have been implemented:

- [x] DQN [[1]](http://arxiv.org/abs/1312.5602)
- [x] DDPG [[2]](http://arxiv.org/abs/1509.02971)
- [x] TD3
- [ ] PPO

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

## References

1. Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning." *arXiv preprint arXiv:1312.5602* (2013).
2. Lillicrap, Timothy P., et al. "Continuous control with deep reinforcement learning." *arXiv preprint arXiv:
   1509.02971* (2015).
3. Fujimoto, Scott, Herke van Hoof, and David Meger. "Addressing function approximation error in actor-critic
   methods." *arXiv preprint arXiv:1802.09477* (2018).
4. Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural
   networks." *Advances in neural information processing systems.* 2012.
5. Goodfellow, Ian, et al. "Generative adversarial nets." *Advances in neural information processing systems.* 2014.
6. Mirza, Mehdi, and Simon Osindero. "Conditional generative adversarial nets." *arXiv preprint arXiv:1411.1784* (2014).
7. Karras, Tero, et al. "Progressive growing of gans for improved quality, stability, and variation." *arXiv preprint
   arXiv:1710.10196* (2017).
8. He, Kaiming, et al. "Deep residual learning for image recognition." *Proceedings of the IEEE conference on computer
   vision and pattern recognition.* 2016.
9. Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image
   segmentation." *International Conference on Medical image computing and computer-assisted intervention.* Springer,
   Cham, 2015.
10. Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." *arXiv
    preprint arXiv:1409.1556* (2014).
