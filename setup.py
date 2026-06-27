from setuptools import setup, find_packages

setup(
    name="amt-yolo",
    version="0.1.0",
    description="AMT-YOLO: Adaptive Memory Trajectory YOLO for Real-Time Video Analysis",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="AMT-YOLO Research Team",
    python_requires=">=3.12",
    packages=find_packages(),
    install_requires=[
        "torch>=2.3.0",
        "torchvision>=0.18.0",
        "ultralytics>=8.2.0",
        "opencv-python>=4.10.0",
        "numpy>=1.26.0",
        "PyYAML>=6.0.1",
        "omegaconf>=2.3.0",
        "einops>=0.8.0",
        "tqdm>=4.66.0",
        "rich>=13.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.2.0",
            "pytest-cov>=5.0.0",
            "black>=24.4.0",
            "isort>=5.13.0",
        ],
        "logging": [
            "wandb>=0.17.0",
            "tensorboard>=2.17.0",
        ],
        "export": [
            "onnx>=1.16.0",
            "onnxruntime-gpu>=1.18.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    entry_points={
        "console_scripts": [
            "amt-train=scripts.train:main",
            "amt-eval=scripts.evaluate:main",
            "amt-demo=scripts.demo:main",
        ],
    },
)
