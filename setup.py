import setuptools

setuptools.setup(
    author="Jihong Ju",
    author_email="daniel.jihong.ju",
    extras_require={
        "test": [
            "codecov",
            "mock",
            "pytest",
            "pytest-cov",
            "pytest-pep8",
            "pytest-runner"
        ],
    },
    install_requires=[
        "scikit-learn",
        "tensorflow",
        "keras"
    ],
    license="MIT",
    name="deep-pu-learn",
    packages=setuptools.find_packages(
        exclude=[
            "experiments",
            "tests"
        ]
    ),
    url="https://github.com/JihongJu/deep-pu-learning",
    version="0.0.1"
)
