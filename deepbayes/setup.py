import pathlib
from setuptools import setup
from setuptools import find_packages, setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="deepbayes",
    version="0.0.1",
    description="Package for doing approximate Bayesian inference on deep neural networks [based in TF 2.0+]",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/deepbayes/deepbayes",
    author="Matthew Wicker",
    author_email="mattrwicker@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    #packages=["deepbayes", "deepbayes.optimizers", "deepbayes.analyzers"],
    packages=find_packages(),
    include_package_data=True,
    install_requires=["tensorflow", "numpy",  "tensorflow-probability", "tqdm"]
)
