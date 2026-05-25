"""Analogical setup."""

from setuptools import find_packages, setup

setup(
    name="analogical",
    version="0.1.0",
    description=(
        "Perception-first, zero-pretraining, real-time adaptive agent "
        "with online adaptation and MuJoCo/mock simulation support."
    ),
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "torch>=2.0",
    ],
    extras_require={
        "mujoco": ["mujoco>=3.0"],
        "dev": ["pytest>=7.0"],
    },
    entry_points={
        "console_scripts": [
            "analogical-benchmark=run_benchmark:main",
        ]
    },
)
