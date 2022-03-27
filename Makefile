
clean:
	rm -rf build dist venv rubikscubetracker.egg-info
	find . -name __pycache__ | xargs rm -rf

init: clean
	python3 -m venv venv
	./venv/bin/python3 -m pip install -U pip==22.0.4
	./venv/bin/python3 -m pip install -r requirements.dev.txt
	./venv/bin/python3 -m pre_commit install --install-hooks --overwrite
	./venv/bin/python3 -m pip check

format:
	isort rubikscubetracker usr test.py setup.py
	@./venv/bin/python3 -m black --config=pyproject.toml .
	@./venv/bin/python3 -m flake8 --config=.flake8

wheel:
	@./venv/bin/python3 setup.py bdist_wheel

test:
	./test.py
