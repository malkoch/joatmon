# joatmon (jack of all trades, master of none)

[![Documentation Status](https://readthedocs.org/projects/joatmon/badge/?version=latest)](https://joatmon.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/joatmon.svg)](https://badge.fury.io/py/joatmon)
[![codecov](https://codecov.io/gh/malkoch/joatmon/branch/master/graph/badge.svg?token=LLMWHT1CN1)](https://codecov.io/gh/malkoch/joatmon)
![GitHub](https://img.shields.io/github/license/malkoch/joatmon)
[![Pylint](https://github.com/malkoch/joatmon/actions/workflows/pylint.yml/badge.svg)](https://github.com/malkoch/joatmon/actions/workflows/pylint.yml)
[![Package](https://github.com/malkoch/joatmon/actions/workflows/package.yml/badge.svg)](https://github.com/malkoch/joatmon/actions/workflows/package.yml)
[![Release](https://github.com/malkoch/joatmon/actions/workflows/release.yml/badge.svg)](https://github.com/malkoch/joatmon/actions/workflows/release.yml)

Welcome to the Joatmon repository! Here you'll find a collection of codes and scripts I've written over the past 6 years, spanning a wide range of topics and applications. From automation and neural networks to OpenAI Gym environments, this repository truly lives up to its name – a jack of all trades.

## Disclaimer

Please exercise caution when using the scripts and modules in this repository. While I've put in a lot of effort to create these resources, they might contain bugs, glitches, or unforeseen issues. It's essential to thoroughly review and test any code you intend to use in your projects. I strongly recommend not using these scripts in production environments without proper testing and validation.

## Table of Contents
    About the Repository
    Usage
    Contributing
    Installation
    Examples
    How to run the tests
    License
    Disclaimer

## About the Repository

This repository is a testament to my journey over the years as I've dabbled in various domains of programming and technology. You'll find scripts, modules, and projects covering a wide spectrum, including:

    Automation scripts to simplify repetitive tasks.
    Neural network implementations for various machine learning experiments.
    OpenAI Gym environments and reinforcement learning projects.
    And much more!

While the repository is a reflection of my curiosity and eagerness to explore different fields, please be aware that the quality and stability of the scripts may vary. Some of the codes might be outdated due to evolving technologies and practices, and others might not be thoroughly tested.

## Usage

To make the best use of this repository:

    Review Code Thoroughly: Carefully review the code and scripts you're interested in before integrating them into your projects. This will help you understand their functionality and potential pitfalls.
    Test Rigorously: Before deploying any code to production or critical systems, conduct thorough testing in controlled environments to identify and rectify any issues.
    Contribute: If you find a bug, want to improve existing code, or have a new script to add, contributions are welcome! Please follow the Contributing guidelines below.

## Contributing

If you'd like to contribute to Joatmon, follow these steps:

    Fork the repository.
    Create a new branch for your changes: git checkout -b feature/your-feature-name.
    Make your modifications and improvements.
    Test your changes rigorously.
    Commit your changes: git commit -m "Add your meaningful commit message here".
    Push to the branch: git push origin feature/your-feature-name.
    Open a Pull Request in this repository, detailing your changes and the motivation behind them.

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

## License

This repository is available under the MIT License. However, please be aware that the license applies to the codebase as it exists at the time of your usage. Some portions of the code might have their own licenses or restrictions. Make sure to review and respect the licensing terms of any third-party libraries or assets used in the scripts.

Thank you for exploring Joatmon! Your caution, curiosity, and contributions are highly appreciated.

## Disclaimer
Use these scripts at your own risk. I am not responsible for any damages, losses, or inconveniences caused by the use of these resources. Always prioritize proper testing and validation in your projects.

You can find more information in the [doc](https://joatmon.readthedocs.io/en/latest/).
