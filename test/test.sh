pipenv run flake8 --config=test/test.cfg
pipenv run mypy --config-file=test/test.cfg
pipenv run python test/test.py