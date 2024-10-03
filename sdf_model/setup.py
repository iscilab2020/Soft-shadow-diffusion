from setuptools import setup

setup(
    name="implicit",
    packages=[
        "implicit",
        "implicit.third_party",
        "implicit.third_party.sdf_vae.models",
    ],
    install_requires=[
        "filelock",
        "Pillow",
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
        "trimesh",
        "open3d",
        "pytorch_lightning",
    ],
    author="ISCI Lab",
)



