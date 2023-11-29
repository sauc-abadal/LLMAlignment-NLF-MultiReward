from setuptools import setup, find_packages

with open("requirements.txt") as fp:
    requirements = fp.read().splitlines()

setup(
    name="ctgnlf",
    version="0.0.0",
    description="A library for controlled text generation using NL feedback",
    author="Saüc Abadal*",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=requirements,
    url="https://github.com/sauc-abadal/MultiTask-CTG-NLF",
)
