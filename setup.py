"""
Setup configuration for DHC-SSM Architecture

Deterministic Hierarchical Causal State Space Model
A revolutionary AI architecture eliminating probabilistic sampling uncertainty.
"""

from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Define requirements
requirements = [
    "torch>=2.0.0",
    "torch-geometric>=2.3.0",
    "numpy>=1.21.0",
    "scipy>=1.8.0",
    "matplotlib>=3.5.0",
    "scikit-learn>=1.1.0",
    "tqdm>=4.64.0",
]

# Development requirements
dev_requirements = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=22.0.0",
    "flake8>=5.0.0",
    "mypy>=0.991",
    "jupyter>=1.0.0",
    "notebook>=6.4.0",
]

setup(
    name="dhc-ssm-architecture",
    version="1.0.0",
    author="Sung Hun Kwag",
    author_email="sunghunkwag@gmail.com",
    description="Deterministic Hierarchical Causal State Space Model - Revolutionary O(n) AI Architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sunghunkwag/DHC-SSM-Architecture",
    project_urls={
        "Bug Reports": "https://github.com/sunghunkwag/DHC-SSM-Architecture/issues",
        "Source": "https://github.com/sunghunkwag/DHC-SSM-Architecture",
        "Documentation": "https://github.com/sunghunkwag/DHC-SSM-Architecture/blob/main/README.md",
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
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "all": requirements + dev_requirements,
    },
    entry_points={
        "console_scripts": [
            "dhc-ssm-demo=examples.dhc_ssm_demo:main",
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
    ],
    include_package_data=True,
    zip_safe=False,
)
