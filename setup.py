from setuptools import setup, find_packages

setup(
    name="lottery",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "matplotlib",
        "seaborn",
        "scipy",
        "scikit-learn",
        "marshmallow>=3.0.0"
    ],
    python_requires=">=3.8",
)