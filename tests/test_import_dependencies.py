import pytest

# List of dependencies
dependencies = [
    "check_distribution",
    "FLiESANN",
    "gedi_canopy_height",
    "GEOS5FP",
    "MODISCI",
    "numpy",
    "rasters"
]

# Generate individual test functions for each dependency
@pytest.mark.parametrize("dependency", dependencies)
def test_dependency_import(dependency):
    __import__(dependency)
