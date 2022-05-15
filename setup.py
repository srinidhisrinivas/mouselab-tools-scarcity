from setuptools import setup

setup(
    name="mouselab",
    version="",
    packages=["mouselab", "mouselab.envs"],
    url="",
    license="",
    author="Fred Callaway, Sayan Gul, Yash Raj Jain",
    author_email="",
    description="",
    setup_requires=["wheel"],
    install_requires=[
        "numpy",
        "scipy",
        "contexttimer",
        "dill",
        "gym",
        "IPython",
        "joblib",
        "keras",
        "matplotlib",
        "more_itertools",
        "networkx",
        "pandas",
        "PyContracts",
        # to solve incompatibility with pyparsing and pycontracts
        "pyparsing>=2.4.0, <3.0.0",
        "seaborn",
        "toolz",
        "tqdm",
        "parameterized",
    ],
)
