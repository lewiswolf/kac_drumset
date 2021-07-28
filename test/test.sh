pipenv run flake8 --config=test/test.cfg index.py src test
pipenv run mypy --config-file=test/test.cfg index.py src test
pipenv run python test/unit_tests.py