"""
Example of how to use setuptools
"""

__version__ = "0.0.1"

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    author='Jeff Wang',
    author_email='jeffwji@test.com',

    name="HelloTPOT",
    version_command='git describe --always --long --dirty=-dev',  # 3) 获得　tag 动态获得版本号(参考文档 <git release flow>)

    packages=find_packages(
        exclude=['tests', '*.tests', '*.tests.*']
    ),

    package_data={
        '': ['config/*.properties', '*.md', 'requirements.txt'],
    },

    install_requires=requirements,
)
