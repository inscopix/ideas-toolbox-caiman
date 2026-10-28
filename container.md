# Caiman Container Image

This container has the necessary python packages for analysis in this repo.

**Base Image:** [python3.13.12](https://hub.docker.com/layers/library/python/3.13.12/images/sha256-5fe01519daaaa2a7808ee1bb7124b958e052a66b5011cabbd6707fceb55b3c95)

## Packages Installed:

The following packages are installed for analysis:

* `ideas-python[analysis]`

### Development

For development, the following packages are installed:

* `pytest`
* `pytest-cov`
* `pytest-mock`
* `pytest-order`
* `pytest-watch`
* `pre-commit`
* `ruff`
