from setuptools import setup, find_packages


def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


reqs = parse_requirements('requirements.txt')


with open('README.md') as f:
    long_description = f.read()


setup(name='pytorch2keras',
      version='0.2.4',
      description='The deep learning models converter',
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
