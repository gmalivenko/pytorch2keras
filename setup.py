from setuptools import setup, find_packages


try:  # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements


# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements('requirements.txt', session='null')


# reqs is a list of requirement
# e.g. ['django==1.5.1', 'mezzanine==1.4.6']
reqs = [str(ir.req) for ir in install_reqs]


with open('README.md') as f:
    long_description = f.read()


setup(name='pytorch2keras',
      version='0.1.11',
      description='The deep learning models convertor',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/nerox8664/pytorch2keras',
      author='Grigory Malivenko',
      author_email='nerox8664@gmail.com',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Image Recognition',
      ],
      keywords='machine-learning deep-learning pytorch keras neuralnetwork vgg resnet '
               'densenet drn dpn darknet squeezenet mobilenet',
      license='MIT',
      packages=find_packages(),
      install_requires=reqs,
      zip_safe=False)
