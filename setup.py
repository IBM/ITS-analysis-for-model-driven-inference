"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
import setuptools
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name="model-driven-inference-its",  # Required
    version="1.0.3",  # Required
    description="Interrupted time series wrapper code based on prophet and Poisson regression",  # Optional
    long_description=long_description,  # Optional
    long_description_content_type="text/markdown",  # Optional (see note above)
    url="https://github.com/IBM/ITS-analysis-for-model-driven-inference",  # Optional
    author="IBM Research Africa",  # Optional
    # author_email="catherine.wanjiru@ibm.com",  # Optional
    classifiers=[  # Optional
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Documentation :: Sphinx",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="Interrupted Time Series, Time Series, Policy Makers",  # Optional
    # When your source code is in a subdirectory under the project root, e.g.
    # `src/`, it is necessary to specify the `package_dir` argument.
    package_dir={"": "src"},  # Optional
    # packages=setuptools.find_packages(where="BaseITS"),  # Required
    packages=["BaseITS"],
    python_requires=">=3.9",
    install_requires=[
        "matplotlib",
        "prophet",
        "numpy",
        "pandas",
        "scipy",
        "scikit_learn",
        "seaborn",
        "statsmodels",
    ],  # Optional
    extras_require={  # Optional
        # "dev": ["check-manifest"],
        "test": ["coverage", "pytest"],
    },
    # package_data={  # Optional
    #     "sample": ["package_data.dat"],
    # },
    # entry_points={  # Optional
    #     "console_scripts": [
    #         "sample=sample:main",
    #     ],
    # },
    project_urls={  # Optional
        "Bug Reports": "https://github.com/IBM/ITS-analysis-for-model-driven-inference/issues",
        # "Funding": "https://donate.pypi.org",
        # "Say Thanks!": "http://saythanks.io/to/example",
        "Source": "https://github.com/IBM/ITS-analysis-for-model-driven-inference/",
    },
)
