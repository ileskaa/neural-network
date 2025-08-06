dev:
	poetry run flask --app src/web/app run --debug

lint:
	poetry run pylint src/model

coverage:
	poetry run bash -c "coverage run --branch -m pytest src && coverage report -m"
