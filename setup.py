from setuptools import setup, find_packages
import os

def read_requirements(path="requirements.txt"):
    if not os.path.exists(path):
        return []
    with open(path, "r") as file:
        return [
            line.strip()
            for line in file
            if line.strip() and not line.strip().startswith("#")
        ]

setup(
    name="tos",
    version="1.0.0",
    packages=find_packages(),
    python_requires='>=3.10',
    author="Pingyue Zhang",
    description="Theory of Space",
    long_description=open("README.md", "r").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    install_requires=read_requirements(),
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
