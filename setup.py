"""
Setup configuration for DHC-SSM Enhanced Architecture v2.0

Deterministic Hierarchical Causal State Space Model
An AI architecture with deterministic learning approach.

v2.0 Features:
- Error handling improvements
- Shape validation
- Configuration management system
- Device consistency improvements
- Testing suite
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = []
    for line in fh:
        line = line.strip()
        if line and not line.startswith("#") and ">=" in line:
            requirements.append(line)

# Development requirements
dev_requirements = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-mock>=3.6.0",
    "black>=22.0.0",
    "flake8>=5.0.0",
    "isort>=5.10.0",
    "mypy>=0.991",
    "jupyter>=1.0.0",
    "notebook>=6.4.0",
    "tensorboard>=2.10.0",
    "pre-commit>=2.15.0"
]

setup(
    name="dhc-ssm-architecture",
    version="2.0.0",
    author="Sung Hun Kwag",
    author_email="sunghunkwag@gmail.com",
    description="DHC-SSM Enhanced v2.0 - Deterministic O(n) AI Architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sunghunkwag/DHC-SSM-Architecture",
    project_urls={
        "Bug Reports": "https://github.com/sunghunkwag/DHC-SSM-Architecture/issues",
        "Source": "https://github.com/sunghunkwag/DHC-SSM-Architecture",
        "Documentation": "https://github.com/sunghunkwag/DHC-SSM-Architecture/blob/main/README.md",
        "Changelog": "https://github.com/sunghunkwag/DHC-SSM-Architecture/releases",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Natural Language :: English",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "all": requirements + dev_requirements,
        "gpu": [
            "torch-scatter>=2.1.0",
            "torch-sparse>=0.6.0",
            "torch-cluster>=1.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dhc-ssm-demo=examples.demo:main",
            "dhc-ssm-legacy-demo=examples.dhc_ssm_demo:main",
        ],
    },
    keywords=[
        "artificial intelligence",
        "machine learning",
        "deep learning",
        "state space models",
        "causal reasoning",
        "deterministic learning",
        "graph neural networks",
        "multi-objective optimization",
        "information theory",
        "pareto optimization",
        "O(n) complexity",
        "transformer alternative",
        "shape validation",
        "configuration management",
    ],
    include_package_data=True,
    zip_safe=False,
    # Package metadata for PyPI
    project_metadata={
        "version": "2.0.0",
        "architecture": "DHC-SSM",
        "complexity": "O(n)",
        "learning_type": "deterministic",
        "probabilistic_sampling": False,
        "status": "beta",
    }
)