#from distutils.core import setup
from setuptools import setup

version = "0.1" # First version

version_str = """# 
__version__ = "{0}"\n""".format(version)

fp = open('wfc3dash/version.py','w')
fp.write(version_str)
fp.close()

setup(name='wfc3dash',
      version=version,
      description='Helper scripts for WFC3 DASH observations',
      author='Gabriel Brammer',
      author_email='gbrammer@gmail.com',
      url='https://github.com/gbrammer/wfc3dash/',
      packages=['wfc3dash', 'wfc3dash/grism'],
     )
