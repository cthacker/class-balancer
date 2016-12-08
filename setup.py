from setuptools import setup


if __name__ == "__main__":
    setup(
        name='balancer',
        version='0.0.1',
        description='Python module for handling unbalanced datasets',
        url='https://github.com/cthacker/class-balancer',
        packages=['balancer'],
        install_requires=['numpy>=1.10.4', 'scikit-learn>=0.17.1']
    )
