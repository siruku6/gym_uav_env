# Making this file, auther refered the following repositories.
# - psf / requests
#     https://github.com/psf/requests
#
# - c60evaporator / seaborn-analyzer
#     https://github.com/c60evaporator/seaborn-analyzer
#
#     This is made by the author of this article.
#     - 【PyPI】Pythonの自作ライブラリをpipに公開する方法
#     https://qiita.com/c60evaporator/items/e1ecccab07a607487dcf

from setuptools import setup


def get_version() -> str:
    version: dict[str, str] = {"__name__": "__not_main__"}
    with open("pip_sample/version.py") as fp:
        exec(fp.read(), version)

    return version["__version__"]


DESCRIPTION = (
    "pip_sample: This shows necessary files and"
    "structure of files for making a pip module."
)
NAME = "pip_sample"
AUTHOR = "siruku6"
AUTHOR_EMAIL = "sirukufarios@gmail.com"
URL = "https://github.com/siruku6/pip_sample"
LICENSE = "MIT License"
DOWNLOAD_URL = "https://github.com/siruku6/pip_sample"
VERSION = get_version()
PYTHON_REQUIRES = ">=3.8"

INSTALL_REQUIRES = [
    "requests>=2.27.0",
]
EXTRAS_REQUIRE = {
    "tutorial": [],
}
TEST_REQUIRES = [
    "pytest>=3",
]

PACKAGES = ["pip_sample"]

CLASSIFIERS = [
    # NOTE: Refer to this link, https://e-tec-memo.herokuapp.com/article/177/
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Framework :: Matplotlib",
]

with open("README.md", "r") as f:
    readme = f.read()

setup(
    name=NAME,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    maintainer=AUTHOR,
    maintainer_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=readme,
    long_description_content_type="text/markdown",
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    tests_require=TEST_REQUIRES,
    packages=PACKAGES,
    classifiers=CLASSIFIERS,
)
