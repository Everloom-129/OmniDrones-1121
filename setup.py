from setuptools import find_packages, setup

setup(
    name="omni_drones",
    version="0.1.1",
    author="btx0424@SUSTech",
    keywords=["robotics", "rl"],
    packages=find_packages("."),
    install_requires=[
        "hydra-core",
        "omegaconf",
        "wandb",
        "imageio",
        "plotly",
        "einops",
        "pandas",
        "moviepy",
        "av",
        "torchrl", # for torch==2.2.2
        "tensordict",
    ],
)
