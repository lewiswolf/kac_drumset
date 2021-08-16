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

In either case, _pytorch_ is installed automatically, and will work fine for all CPU based usages. However, to configure this application for GPU usage, you must reinstall the appropriate version of _pytorch_ for your machine (which can be found [here](https://pytorch.org/get-started/locally/)) via:

```bash
$ pipenv run pip install torch==1.8.1+cu102 ...
```

To ensure that the GPU can be fully utilised by this application, make sure to update the _PATH_2_CUDA_ variable in `src/settings.py`, which should point to your installed version of the CUDA SDK.

## Testing

_For development only._

This command first lints the project, and then ensures that all types are correctly handled throughout the codebase. Finally, any unit tests housed in `test/unit_tests.py` are executed.

```
$ pipenv run test
```

## Generate Dataset

When training and evaluating this project, it is necessary to generate a dataset from scratch via:

```bash
$ pipenv run generate
```

This command will generate a dataset, which is stored in `/data`. These files will be accessed and reused by subsequent usages of this application. The settings for this dataset can be configured in `/src/settings.py`. Changes made to these settings may require the dataset to be reconstructed from scratch, however most changes will only need minor ammendation to the original files.

## Train Model

Not yet implemented...

```bash
$ pipenv run train
```

## Train Model

Not yet implemented...

```bash
$ pipenv run evaluate
```

## Working with LaTeX

The paper associated with this project is contained in `/paper`, and can _only_ be updated when the dev packages are installed. The template used for this document can be found [here](https://github.com/lewiswolf/personal-latex-template.git), and is designed for use with one of latex's command line interfaces such as [MacTeX](https://formulae.brew.sh/cask/mactex-no-gui). For convenience, this project is packaged with a helper method that will reconstruct the pdf document from the raw `.tex` files, via:

```bash
$ pipenv run latex
```
