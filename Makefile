PLATFORM=linux/amd64

# Label may be specified in codebuild pipeline
# Locally, use default "latest"
ifndef LABEL
	LABEL=latest
endif

IMAGE_TAG := platform/caiman:${LABEL}
CONTAINER_NAME := ideas-toolbox-caiman

PYTHON=python3.10

.PHONY: help build test clean

.DEFAULT_GOAL := build

build:
	docker build . -t $(IMAGE_TAG) \
		--platform ${PLATFORM} \
		--target base

test: build clean 
	@echo "Running toolbox tests..."
	-mkdir -p $(PWD)/outputs
	docker run \
		--platform ${PLATFORM} \
		-v $(PWD)/data:/ideas/data \
		-v $(PWD)/toolbox:/ideas/toolbox \
		-w /ideas --rm \
		--name $(CONTAINER_NAME) \
		${IMAGE_TAG} \
		$(PYTHON) -m pytest $(TEST_ARGS)

