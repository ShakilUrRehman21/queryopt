.PHONY: test lint format clean cli

test:
	py -m pytest -v

lint:
	py -m ruff check packages tests

format:
	py -m ruff format packages tests

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.pytest_cache')]"

cli:
	py -m queryopt.cli --help
