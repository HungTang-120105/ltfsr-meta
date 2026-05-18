from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ltfsr-meta",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Long-tail Few-shot Recognition with Meta-Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ltfsr-meta",
    packages=find_packages(exclude=["notebooks", "results", "data"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.23.0",
        "pandas>=1.5.0",
        "Pillow>=9.0.0",
        "scipy>=1.10.0",
        "pytorch-lightning>=2.0.0",
        "timm>=0.9.0",
        "scikit-learn>=1.3.0",
        "tensorboard>=2.13.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "notebook>=6.5.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
    },
)
