from setuptools import setup, find_packages

# Open the README file
with open('readme.md', 'r') as f:
    long_description = f.read()

# Open the requirements file
with open('requirements.txt', 'r') as f:
    requirements = f.readlines()

setup(
    name='tmll',
    version='0.0.30',
    packages=find_packages(include=['tmll', 'tmll.*']),
    install_requires=requirements,
    long_description=long_description,
    long_description_content_type='text/markdown',
)
