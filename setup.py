from setuptools import setup, find_packages

with open('readme.md', 'r') as f:
    long_description = f.read()

setup(
    name='tmll',
    version='0.0.9',
    packages=find_packages(include=['tmll', 'tmll.*']),
    install_requires=['requests==2.31.0',
                      'pandas==2.1.4',
                      'numpy==1.26.3',
                      'scikit_learn==1.4.2',
                      'scipy==1.11.4',
                      'tqdm==4.66.2',
                      ],
    long_description=long_description,
    long_description_content_type='text/markdown',
)
