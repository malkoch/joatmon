from __future__ import print_function

import os
import platform
import sys
from distutils.core import setup

from setuptools import find_packages

from joatmon import VERSION


extras = {
    'ai': [
        'numpy'
    ],
    'analysis': [
        'numpy',
        'pandas'
    ],
    'array': [
        'transitions'
    ],
    'assistant': [],
    'decorator': [
        'async_exit_stack'
    ],
    'game': [
        'gym',
        'opencv-python',
        'pygame',
        'pymunk'
    ],
    'image': [
        'opencv-python'
    ],
    'nn': [
        'numpy',
        'six'
    ],
    'orm': [],
    'plugin': [
        'pyjwt',
        'redis',
        'couchbase',
        'elasticsearch',
        'pymongo',
        'psycopg2',
        'confluent_kafka',
        'pika',
        'pyotp'
    ],
    'structure': [],
    'system': [
        'opencv-python',
        'SpeechRecognition',
        'numpy'
    ]
}  # plugin requirements
if sys.platform == 'win32':
    extras['system'].extend(['pywin32'])
extras['all'] = list(set([item for group in extras.values() for item in group]))

if sys.version_info < (3,):
    print('Python 2 has reached end-of-life and is no longer supported by joatmon.')
    sys.exit(-1)
if sys.platform == 'win32' and sys.maxsize.bit_length() == 31:
    print('32-bit Windows Python runtime is not supported. Please switch to 64-bit Python.')
    sys.exit(-1)

python_min_version = (3, 9, 1)
python_min_version_str = '.'.join(map(str, python_min_version))
if sys.version_info < python_min_version:
    print('You are using Python {}. Python >={} is required.'.format(platform.python_version(), python_min_version_str))
    sys.exit(-1)

if __name__ == '__main__':
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.md'), encoding='utf-8') as f:
        long_description = f.read()

    version_range_max = max(sys.version_info[1], 8) + 1
    setup(
        name=os.getenv('JOATMON_PACKAGE_NAME', 'joatmon'),
        version=VERSION,
        description='General Purpose Python Library',
        long_description=long_description,
        long_description_content_type='text/markdown',
        packages=[package for package in find_packages(exclude='tests')],
        # entry_points={
        #     'console_scripts': [
        #         'assistant = scripts.assistant:main'
        #     ]
        # },
        install_requires=[
            'bumpversion',
            'mkgendocs'
        ],
        tests_require=[
            'pytest',
        ],
        extras_require=extras,
        package_data={
            'joatmon': [
                'game/assets/chess/*.png',
                'game/assets/sokoban/sprites/*.png',
                'game/assets/sokoban/xmls/*.xml',
                'assistant/assets/*.wav'
            ]
        },
        url='https://github.com/malkoch/joatmon',
        download_url='https://github.com/malkoch/joatmon/tags',
        author='Hamitcan Malkoç',
        author_email='hamitcanmalkoc@gmail.com',
        python_requires='>={}'.format(python_min_version_str),
        classifiers=[
                        'Development Status :: 5 - Production/Stable',
                        'Intended Audience :: Developers',
                        'Topic :: Software Development',
                        'Topic :: Software Development :: Libraries',
                        'Topic :: Software Development :: Libraries :: Python Modules',
                        'Programming Language :: Python :: 3',
                    ] + ['Programming Language :: Python :: 3.{}'.format(i) for i in range(python_min_version[1], version_range_max)],
        license='MIT',
        keywords='joatmon ml idm automation iva',
    )
