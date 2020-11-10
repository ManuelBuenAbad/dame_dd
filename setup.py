from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='dame_dd',
      version='0.1',
      description='some words for description',
      long_description=readme(),
      url='http://github.com/buenabad/dame_dd',
      author='Manuel Buen-Abad',
      author_email='manuelbuenabadnajar@gmail.com',
      license='Brown',
      packages=['dame_dd'],
      install_requires=[
          'numpy'
      ],
      zip_safe=False)
