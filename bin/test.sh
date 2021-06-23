# arg is used to stop mypy getting annoyed about the src folder...
# could/should probably be deleted in the future
pipenv run flake8 --max-line-length 110 --ignore W191,E261,E402 index.py
pipenv run mypy --ignore-missing-imports index.py 