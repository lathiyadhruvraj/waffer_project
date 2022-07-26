from setuptools import setup,find_packages
from typing import List

#Declaring variables for setup functions
PROJECT_NAME="waffer-predictor"
VERSION="0.0.1"
AUTHOR="Dhruvraj Lathiya"
DESRCIPTION="This is my first project using mlops"
REQUIREMENT_FILE_NAME="requirements.txt"
HYPHEN_E_DOT = "-e ."

with open("README.md", "r", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

def get_requirements_list() -> List[str]:
    """
    Description: This function is going to return list of requirement
    mention in requirements.txt file
    return This function is going to return a list which contain name
    of libraries mentioned in requirements.txt file
    """
    with open(REQUIREMENT_FILE_NAME) as requirement_file:
        requirement_list = requirement_file.readlines()
        requirement_list = [requirement_name.replace("\n", "") for requirement_name in requirement_list]
        if HYPHEN_E_DOT in requirement_list:
            requirement_list.remove(HYPHEN_E_DOT)
        return requirement_list



setup(
    name=PROJECT_NAME,
    version=VERSION,
    author=AUTHOR,
    description=DESRCIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
    url="https://github.com/lathiyadhruvraj/waffer_project",
    author_email="lathiyadhruvraj44@gmail.com",
    license="GNU",
    packages=["src"],
    install_requires=get_requirements_list()
)



 
    
