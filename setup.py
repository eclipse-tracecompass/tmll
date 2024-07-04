from setuptools import setup, find_packages

setup(
    name='tmll',
    version='0.0.5',
    packages=find_packages(include=['tmll', 'tmll.*']),
    install_requires=['requests==2.31.0',
                      'pandas==2.1.4',
                      'numpy==1.26.3',
                      'scikit_learn==1.4.2',
                      'scipy==1.11.4',
                      'tqdm==4.66.2',
                      ]
)
