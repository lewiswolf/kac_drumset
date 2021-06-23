## Dependencies

-   [pipenv](https://formulae.brew.sh/formula/pipenv#default)

## Install

To run this project:

```bash
$ pipenv install
```

or to install this project for development:

```bash
$ pipenv install -d
```

## Test

_For development only._

```
$ pipenv run test
```

## Generate Dataset

```bash
$ pipenv run generate
```

## Train Model

```bash
pipenv run train
```

<!-- Pytorch is installed automatically, and will work fine for all CPU based usages. However, to configure this package for GPU usage, you must install your required pytorch version via:

```bash
$ pipenv run pip install torch==1.8.1+cu102 ...
``` -->
