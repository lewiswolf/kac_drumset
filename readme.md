## Dependencies

-   [pipenv](https://formulae.brew.sh/formula/pipenv#default)
-   [CUDA SDK](https://developer.nvidia.com/cuda-downloads)

## Install

To run this project:

```bash
$ pipenv install
```

or to install this project for development:

```bash
$ pipenv install -d
```

In either case, _pytorch_ is installed automatically, and will work fine for all CPU based usages. However, to configure this package for GPU usage, you must reinstall the appropriate version of _pytorch_ for your machine (which can be found [here](https://pytorch.org/get-started/locally/)) via:

```bash
$ pipenv run pip install torch==1.8.1+cu102 ...
```

To ensure that the GPU can be fully utilised by this application, make sure to update the _PATH_2_CUDA_ variable in `src/settings.py`, which should point to your installed version of the CUDA SDK.

## Test

_For development only._

```
$ pipenv run test
```

## Generate Dataset

```bash
$ pipenv run generate
```

Not yet implemented...

## Train Model

```bash
pipenv run train
```

Not yet implemented...
