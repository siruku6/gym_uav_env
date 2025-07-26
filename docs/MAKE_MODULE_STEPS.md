# Structure of files

```markdown
./  # root directory
 ├─ LICENSE
 ├─ module_name/
 │   ├─ ****.py
 │   :
 │   :
 │   └─ __init__.py
 ├─ README.md
 └─ setup.py
```


# Steps for making module

## 0. prepare files

Make files and directories like this repository and like above `Structure of files`.


## 1. setup necessary libraries

    ```bash
    $ pip install setuptools twine wheel
    ```


## 2. build packages from your source

```bash
$ python3 setup.py sdist
$ python3 setup.py bdist_wheel
```


## 3. make account on PyPI

Make your account on the following pages.

- test page  
https://test.pypi.org/account/register/
- production page  
https://pypi.org/account/register/


## 4. setup config file of twine for PyPI
```bash
$ touch ~/.pypirc
```

### `~/.pypirc`

```config
[distutils]
index-servers =
  pypi
  testpypi

[pypi]
repository: https://upload.pypi.org/legacy/
username: username for account of PyPI
password: password for account of PyPI

[testpypi]
repository: https://test.pypi.org/legacy/
username: username for test account of PyPI
password: password for test account of PyPI
```

You can use API-token substituting username and password by the following [link](https://pypi.org/help/#apitoken).


## 5. upload to PyPI (TEST)

```bash
$ twine upload --repository testpypi dist/*
```


## 6. test to install uploaded package

```bash
$ pip install --index-url https://test.pypi.org/simple/ [package-name]
$ python3
>>> import [package-name]
>>> ... (use your package)
```

You may use your packege like above.

```bash
# after your test is finished
$ pip uninstall [package-name]
```


## 7. upload to PyPI (PRODUCTION)

```bash
$ twine upload --repository pypi dist/*

# You may install your module.
$ pip install [package-name]
```
