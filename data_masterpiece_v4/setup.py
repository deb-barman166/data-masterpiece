from setuptools import setup, find_packages

setup(
    name="data_masterpiece_v4",
    version="4.0.0",
    author="Data Masterpiece Team",
    author_email="contact@datamasterpiece.io",
    description="God-Level Data Science Pipeline — Auto + Manual Mode with CLI",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/datamasterpiece/data_masterpiece_v4",
    packages=find_packages(exclude=["examples", "tests"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "click>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "pytorch": [
            "torch>=1.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dm4=data_masterpiece_v4.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
