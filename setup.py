from setuptools import setup

setup(
    name="occluder",
    packages=[
        "occluder",
        "occluder.datasets",
        "occluder.models",
        "occluder.utils",
    ],
    install_requires=[
        "filelock",
        "Pillow",
        "torch",
        "fire",
        "humanize",
        "requests",
        "tqdm",
        "matplotlib",
        "scikit-image",
        "scipy",
        "numpy",
        "ipython",
        "accelerate",
        "einops",
        "scikit-learn",
        "h5py",
        "pandas",
        "wandb",
        "torch",
        "torchaudio",
        "torchvideo",
        "trimesh",
        "open3d",
    ],
    author="ISCI Lab",
)



