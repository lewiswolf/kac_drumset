#!/bin/bash
# Run formatter, linter and unit tests.

pipenv run flake8
pipenv run mypy
pipenv run python test/test.py