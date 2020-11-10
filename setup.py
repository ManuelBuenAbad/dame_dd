from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='dame_dd',
      version='0.1',
      url='http://github.com/ManuelBuenAbad/dame_dd',
      description="A python code to compute the impact of dark matter substructure on direct detection efforts via electron scattering.",
      long_description=readme(),
      author='Manuel Buen-Abad',
      author_email='manuelbuenabadnajar@gmail.com',
      license='MIT',
      packages=['dame_dd'],
      install_requires=[
          'numpy'
      ],
      zip_safe=False)
