from setuptools import setup, find_packages

setup(
    name="dl-assignment1",
    version="0.1.0",
    description="CO3133 Deep Learning Assignment 1 — Classification on Image, Text, and Multimodal Data",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
)
