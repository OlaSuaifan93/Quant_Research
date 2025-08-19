from setuptools import find_packages,setup
from typing import List


HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    This function will return the requirements
    '''
    requirements=[]
    #with open('requirements.txt') or
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n"," ")for req in requirements]

        if HYPEN_E_DOT in requirements: #we do this because -e . shouldnt come it just triggers setup.py but its not a library to install.
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
name='Quant_Research',
version='0.0.1',
author='olasuaifan',
author_email='suaifanola@gmail.com',
packages=find_packages(),
#install_requires=['pandas','numpy','seaborn']
install_requires=get_requirements('requirements.txt')

)