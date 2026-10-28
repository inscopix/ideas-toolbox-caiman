.PHONY:  clean clean-venv validate-ideas-cli venv set-hooks setup build security test ruff ruff-check run run-all run-all-clean

# Docker image build constants
IMAGE_REPO := platform
IMAGE_NAME := caiman
LABEL := $(shell jq -r '.[0].label' .ideas/images_spec.json)
IMAGE_TAG := ${IMAGE_REPO}/${IMAGE_NAME}:${LABEL}
LATEST_IMAGE_TAG := ${IMAGE_REPO}/${IMAGE_NAME}:latest
PLATFORM := linux/amd64
TARGET ?= final
DOCKER_BUILD = docker build . \
	--platform $(PLATFORM) \
	-t $(LATEST_IMAGE_TAG)

# IDEAS CLI is used to execute tools locally
# It's installed in the project venv, but can be overwritten to a customized local version instead.
IDEAS_CLI ?= .venv/bin/ideas

# Update the tool specs whenever a new version of a container image is created
TOOL_SPECS := $(wildcard .ideas/*/tool_spec.json)

# Determine if there's a lockfile to track changes from
LOCKFILE := $(wildcard uv.lock)

.DEFAULT_GOAL := build

clean:
	@echo "Cleaning up"
	-docker rmi ${IMAGE_TAG}
	-docker rmi ${IMAGE_TAG}-test

clean-venv:
	rm -rf .venv

validate-ideas-cli:
	@test -f ${IDEAS_CLI} || { echo "Error: IDEAS CLI not found. Ensure you have run 'make venv', or override and use your own install of the cli by specifying a value for IDEAS_CLI when executing make commands"; exit 1; }

.venv/touchfile: .python-version pyproject.toml $(LOCKFILE)
	uv sync --no-install-project --only-group dev
	touch $@

venv: .venv/touchfile

set-hooks: venv .pre-commit-config.yaml
	@echo "Installing pre-commit hooks"
	uv run pre-commit install

setup: venv set-hooks

# Builds docker image
# Installs necessary software dependencies for source code
build:
	$(DOCKER_BUILD) \
		--target ${TARGET}
	docker tag ${LATEST_IMAGE_TAG} ${IMAGE_TAG}
	@$(foreach f,$(TOOL_SPECS),\
	jq --indent 4 '.container_image.label = "$(LABEL)"' $(f) > $(f).tmp && \
	mv $(f).tmp $(f);)

# Builds docker image checking security vulnerabilities
security:
	$(DOCKER_BUILD) \
		--target final

# Runs unit tests in docker image
# Used in automated pr checks on github
test: TARGET=test
test: IMAGE_TAG=${IMAGE_REPO}/${IMAGE_NAME}:${LABEL}-test
test: LATEST_IMAGE_TAG=${IMAGE_REPO}/${IMAGE_NAME}:latest-test
test: build
	@echo "Running tests..."
	docker run \
		--platform ${PLATFORM} \
		--rm \
		${IMAGE_TAG} \
		/ideas/venv/bin/python -m pytest ${TEST_ARGS}

# Applies linter on source code
ruff: venv
	@echo "Running ruff..."
	uv run ruff format . $(ARGS)
	uv run ruff check --fix . $(ARGS)

# Checks code formatting of source code
# Used in automated pr checks on github
# Does not actually apply any linter changes on the source code,
ruff-check: venv
	@echo "Running lint..."
	uv run ruff format --check . $(ARGS)
	uv run ruff check --no-fix . $(ARGS)

# Run a tool in the repo
# Specify the tool key to run
run: venv validate-ideas-cli build
	${IDEAS_CLI} tools run $(tool) -s -c -n

# Build docker image, ensuring venv is up-to-date
build-all: venv build

# Test tools, ensuring venv is up-to-date
test-all: venv build
test-all: TARGET=test
test-all: IMAGE_TAG=${IMAGE_REPO}/${IMAGE_NAME}:${LABEL}-test
test-all: LATEST_IMAGE_TAG=${IMAGE_REPO}/${IMAGE_NAME}:latest-test
test-all: test

# Run all tools in the repo consecutively
run-all: venv validate-ideas-cli build
	@set -e; \
	for f in .ideas/*/; do \
		echo "============================================================="; \
		${IDEAS_CLI} tools run -s -c -n "$$(basename "$$f")"; \
	done

run-all-clean: clean-venv venv run-all
