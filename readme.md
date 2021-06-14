## Dependencies

-   [cmake](https://formulae.brew.sh/formula/cmake)
-   [pipenv](https://formulae.brew.sh/formula/pipenv#default)

## Install

```bash
pipenv run install
```

Please note, this is not the typical `pipenv install` command, as this command will also install any necessary C++ libraries, as well as build the project.

## Rebuild C++

```bash
pipenv run buildcpp
```

## Run Standalone

```bash
pipenv run cpp
```

## Run Python (coming soon...)

```bash
pipenv run start
```
