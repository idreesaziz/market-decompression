from setuptools import setup, find_packages

setup(
    name="scalar_reconstruction",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.11",
        "scikit-learn>=1.3",
    ],
    description="Transfer field estimation pipeline for market decompression (Aziz, 2026)",
)
