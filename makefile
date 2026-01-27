# Define the package name, environment name, and Python version
PACKAGE_NAME = BESS-JPL
MODULE_NAME = $(subst -,_,$(PACKAGE_NAME))
ENVIRONMENT_NAME = $(PACKAGE_NAME)
DOCKER_IMAGE_NAME = $(shell echo $(PACKAGE_NAME) | tr '[:upper:]' '[:lower:]')
PYTHON_VERSION = $(if $(PYTHON),$(PYTHON),3.13)

# Clean up build artifacts and temporary files
clean:
	# Remove object files, output files, and log files
	rm -rf *.o *.out *.log
	# Remove build directory
	rm -rf build/
	# Remove distribution directory
	rm -rf dist/
	# Remove egg-info directory
	rm -rf *.egg-info
	# Remove pytest cache
	rm -rf .pytest_cache
	# Find and remove all __pycache__ directories
	find . -type d -name "__pycache__" -exec rm -rf {} +

# Run tests using pytest
test:
	# Execute all tests with pytest in verbose mode
	pytest -vv

# Verify the package installation and functionality
verify:
	# Run verification checks
	python -c "from $(MODULE_NAME).verify import verify; exit(0 if verify() else 1)"

# Build the Python package
build:
	# Use Python's build module to create the package
	python -m build

# Upload the built package to PyPI using Twine
twine-upload:
	# Upload all files in the dist directory to PyPI
	twine upload dist/*

# Clean, build, and upload the package
dist:
	# Clean the workspace
	make clean
	# Build the package
	make build
	# Upload the package to PyPI
	make twine-upload

# Install the package in editable mode with development dependencies
install:
	# Install the package in editable mode with development extras
	pip install -e .[dev]

# Uninstall the package
uninstall:
	# Uninstall the package by name
	pip uninstall $(PACKAGE_NAME)

# Reinstall the package (uninstall and then install)
reinstall:
	# Uninstall the package
	make uninstall
	# Install the package
	make install

# Create a new Conda environment with the specified Python version
environment:
	# Create a Conda environment with the specified name and Python version
	mamba create -y -n $(ENVIRONMENT_NAME) -c conda-forge python=$(PYTHON_VERSION)

# Generate input dataset
generate-input-dataset:
	# Generate input dataset using the package's generator
	python -c "from $(MODULE_NAME).generate_input_dataset import generate_input_dataset; generate_input_dataset()"

# Generate output dataset
generate-output-dataset:
	# Generate output dataset using the package's generator
	python -c "from $(MODULE_NAME).generate_output_dataset import main; main()"

# Generate GEOS5FP inputs
generate-GEOS5FP-inputs:
	# Generate GEOS5FP inputs using the package's generator
	python -c "from $(MODULE_NAME).generate_GEOS5FP_inputs import generate_GEOS5FP_inputs; generate_GEOS5FP_inputs()"

# Remove the Conda environment
remove-environment:
	# Remove the Conda environment by name
	mamba env remove -y -n $(ENVIRONMENT_NAME)

# Start Colima with specific resource settings
colima-start:
	# Start Colima with 16GB memory, x86_64 architecture, and 100GB disk space
	colima start -m 16 -a x86_64 -d 100 

# Build the Docker image
docker-build:
	# Build the Docker image with the latest tag
	docker build -t $(DOCKER_IMAGE_NAME):latest .

# Build the Docker image up to the 'environment' target stage
docker-build-environment:
	# Build the Docker image up to the 'environment' stage with the latest tag
	docker build --target environment -t $(DOCKER_IMAGE_NAME):latest .

# Build the Docker image up to the 'installation' target stage
docker-build-installation:
	# Build the Docker image up to the 'installation' stage with the latest tag
	docker build --target installation -t $(DOCKER_IMAGE_NAME):latest .

# Run the Docker container interactively with the Fish shell
docker-interactive:
	# Run the Docker container interactively and start the Fish shell
	docker run -it $(DOCKER_IMAGE_NAME) fish 

# Remove the Docker image
docker-remove:
	# Forcefully remove the Docker image by name
	docker rmi -f $(DOCKER_IMAGE_NAME)

# Compile the ATBD and User Guide from Markdown to PDF
compile-documentation:
	# Run the shell script to compile the ATBD to PDF
	bash documentation/ECOv003_L3_L4_JET_ATBD.sh
	# Run the shell script to compile the User Guide to PDF
	bash documentation/ECOv003_L3_L4_JET_User_Guide.sh
