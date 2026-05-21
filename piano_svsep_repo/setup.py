from setuptools import setup, find_packages

# try:
#     import torch
# except ImportError:
#     raise ImportError("Pytorch is not installed. Please install it using the proper configurations for your system https://pytorch.org/get-started/locally/")


setup(
    name="piano_svsep",
    version="0.0.1dev",
    packages=find_packages(),
    setup_requires=["torch"],
    install_requires=[
        "torch",
        "torch_geometric",
        "partitura",
        "torchmetrics",
        "scipy",
        "scikit-learn",
        "pytorch_lightning",
        "verovio",
        "joblib",
        "gitpython",
        "tqdm",
    ],
)
