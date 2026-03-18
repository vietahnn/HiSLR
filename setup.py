from setuptools import setup, find_packages

setup(
    name="hislr",
    version="1.0.0",
    description="HiSLR: Hierarchical Multimodal Fusion Network with Phonological-Aware "
                "Pre-Training for Isolated Sign Language Recognition",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "timm>=0.9.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "data": [
            "decord>=0.6.0",
            "opencv-python>=4.7.0",
            "mediapipe>=0.10.0",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
        ],
    },
    entry_points={
        "console_scripts": [
            "hislr-train=hislr.train:main",
            "hislr-eval=hislr.evaluate:main",
        ],
    },
)
