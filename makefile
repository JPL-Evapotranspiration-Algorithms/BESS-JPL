PACKAGE_NAME = BESS-JPL
ENVIRONMENT_NAME = $(PACKAGE_NAME)
DOCKER_IMAGE_NAME = $(PACKAGE_NAME)

clean:
	rm -rf *.o *.out *.log
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +

test:
	pytest -vv

verify:
	python -c "from BESS_JPL.verify import verify; exit(0 if verify() else 1)"

build:
	python -m build

twine-upload:
	twine upload dist/*

dist:
	make clean
	make build
	make twine-upload

remove-environment:
	mamba env remove -y -n $(ENVIRONMENT_NAME)

install:
	pip install -e .[dev]

uninstall:
	pip uninstall $(PACKAGE_NAME)

reinstall:
	make uninstall
	make install

generate-input-dataset:
	python -c "from BESS_JPL.generate_input_dataset import generate_input_dataset; generate_input_dataset()"

generate-output-dataset:
	python -c "from BESS_JPL.generate_output_dataset import main; main()"

generate-GEOS5FP-inputs:
	python -c "from BESS_JPL.generate_BESS_GEOS5FP_inputs import generate_BESS_GEOS5FP_inputs; generate_BESS_GEOS5FP_inputs()"

environment:
	mamba create -y -n $(ENVIRONMENT_NAME) -c conda-forge python=3.11

colima-start:
	colima start -m 16 -a x86_64 -d 100 

docker-build:
	docker build -t $(DOCKER_IMAGE_NAME):latest .

docker-build-environment:
	docker build --target environment -t $(DOCKER_IMAGE_NAME):latest .

docker-build-installation:
	docker build --target installation -t $(DOCKER_IMAGE_NAME):latest .

docker-interactive:
	docker run -it $(DOCKER_IMAGE_NAME) fish 

docker-remove:
	docker rmi -f $(DOCKER_IMAGE_NAME)
