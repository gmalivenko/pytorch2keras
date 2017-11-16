from setuptools import setup, find_packages
from pip.req import parse_requirements

# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements('requirements.txt', session='null')

# reqs is a list of requirement
# e.g. ['django==1.5.1', 'mezzanine==1.4.6']
reqs = [str(ir.req) for ir in install_reqs]

setup(name='pytorch2keras',
      version='0.1',
      description='The model convertor',
      url='',
      author='Grigory Malivenko',
      author_email='nerox8664@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=reqs,
      zip_safe=False)
