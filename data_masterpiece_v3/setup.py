from setuptools import setup, find_packages

setup(
    name="data_masterpiece_v3",
    version="3.0.0",
    description="Legend-Level Python Data Science Pipeline — Auto + Manual Mode",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Data Masterpiece",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scipy>=1.9.0",
        "scikit-learn>=1.1.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "openpyxl>=3.0.0",
    ],
    extras_require={
        "pytorch":    ["torch>=2.0.0"],
        "tensorflow": ["tensorflow>=2.12.0"],
        "all":        ["torch>=2.0.0", "pyarrow>=10.0.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
