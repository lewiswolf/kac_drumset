pipenv run flake8 --max-line-length 110 --ignore W191,E261,E402 index.py src

# --ignore-missing-imports is used to stop mypy getting annoyed at dependencies..
# could/should probably be deleted in the future (may be to do with pipenv)
pipenv run mypy --ignore-missing-imports index.py src