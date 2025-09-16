#!/usr/bin/env python3
"""Setup script for multimodal-agent-framework"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="multimodal-agent-framework",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A multi-modal agent framework with unified interfaces for different AI model providers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/multimodal-agent-framework",
    packages=find_packages(exclude=["tests*", "examples*", "venv*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=5.0",
            "mypy>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # Add any CLI commands here if needed
        ],
    },
    include_package_data=True,
    zip_safe=False,
)