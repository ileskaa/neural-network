dev:
	poetry run flask --app src/web/app run --debug

lint:
	poetry run pylint src/model
