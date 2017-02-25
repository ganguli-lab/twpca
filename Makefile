all:
	python setup.py install

develop:
	python setup.py develop

test:
	pytest --cov=twpca --cov-report=html tests/

lint:
	flake8 twpca/

clean:
	rm -rf htmlcov/
	rm -rf twpca.egg-info
	rm -f twpca/*.pyc
	rm -rf twpca/__pycache__

upload:
	python setup.py sdist upload
