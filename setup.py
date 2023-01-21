from setuptools import setup

setup(
    name='flac-py',
    version='0.0.1',
    url='',
    license='AGPL-3.0-only',

    author='Tancredi Orlando',
    author_email='tancredi.orlando@gmail.com',

    description='',
    long_description='',

    packages=['flac'],

    python_requires='>3.10',

    extras_require={
        'dev': [
            'mypy==0.991',
            'flake8==5.0.4',
            'pytest==7.2.0'
        ]
    },

    entry_points={
        'console_scripts': [
            'flac-py = flac.__main__:main'
        ]
    }
)
