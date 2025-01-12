import os
from setuptools import setup, find_namespace_packages


def readlines(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.readlines()


install_requires = readlines('requirements.txt')


setup(
    name='yolov4-csp',
    version='1.3.2',
    install_requires=install_requires,
    packages=find_namespace_packages(include=['yolo', 'yolo.*']),
    include_package_data=True,
    python_requires='>=3.7'
)
