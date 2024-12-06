# toolbox variables
REPO=inscopix
PROJECT=ideas
MODULE=toolbox
IMAGE_NAME=caiman
VERSION=$(shell git describe --tags --always --dirty)
IMAGE_TAG=${REPO}/${PROJECT}/${MODULE}/${IMAGE_NAME}:${VERSION}
FULL_NAME=${REPO}/${PROJECT}/${MODULE}/${IMAGE_NAME}
CONTAINER_NAME=${REPO}-${PROJECT}-${MODULE}-${IMAGE_NAME}-${VERSION}
PLATFORM=linux/amd64

# this flag determines whether files should be 
# dynamically renamed (if possible) after function 
# execution. 
# You want to leave this to true so that static 
# filenames are generated, so that these can be 
# annotated by the app. 
# If you want to see what happens on IDEAS, you can
# switch this to false 
ifndef TC_NO_RENAME
	TC_NO_RENAME="true"
endif

define run_command
    bash -c 'mkdir -p "/ideas/outputs/$1" \
        && cd "/ideas/outputs/$1" \
        && cp "/ideas/inputs/$1.json" "/ideas/outputs/$1/inputs.json" \
        && "/ideas/commands/$1.sh" \
	    && rm "/ideas/outputs/$1/inputs.json"'
endef

.PHONY: help build test clean

.DEFAULT_GOAL := build

clean:
	@echo "Cleaning up"
	-docker rm $(CONTAINER_NAME)
	-docker images | grep $(FULL_NAME) | awk '{print $$1 ":" $$2}' | grep -v $(VERSION) | xargs docker rmi

build:
	@PACKAGE_REQS=$$(if [ -f ../.dev_requirements.txt ]; then cat ../.dev_requirements.txt | grep -v "#" | tr '\n' ' '; else echo "ideas-public-python-utils@git+https://@github.com/inscopix/ideas-public-python-utils.git@0.0.17 caiman@git+https://github.com/inscopix/CaImAn.git@v0.0.6 isx==2.0.0"; fi) && \
	echo "Building docker image with PACKAGE_REQS: $$PACKAGE_REQS" && \
	DOCKER_BUILDKIT=1 docker build . -t $(IMAGE_TAG) \
		--platform ${PLATFORM} \
		--build-arg PACKAGE_REQS="$$PACKAGE_REQS" \
		--target base

test: build clean 
	@echo "Running toolbox tests..."
	-mkdir -p $(PWD)/outputs
	docker run \
		--platform ${PLATFORM} \
		-v $(PWD)/data:/ideas/data \
		-v $(PWD)/outputs:/ideas/outputs \
		-v $(PWD)/inputs:/ideas/inputs \
		-v $(PWD)/commands:/ideas/commands \
		-w /ideas \
		--name $(CONTAINER_NAME) \
		${IMAGE_TAG} \
		pytest $(TEST_ARGS)

run: build clean
	@bash check_tool.sh $(TOOL)
	@echo "Running the  $(TOOL) tool in a Docker container. Outputs will be in /outputs/$(TOOL)"
	-rm -rf $(PWD)/outputs/
	docker run \
			--platform ${PLATFORM} \
			-v $(PWD)/data:/ideas/data \
			-v $(PWD)/inputs:/ideas/inputs \
			-v $(PWD)/commands:/ideas/commands \
			-e TC_NO_RENAME=$(TC_NO_RENAME) \
			--name $(CONTAINER_NAME) \
	    $(IMAGE_TAG) \
		$(call run_command,$(TOOL)) \
	&& docker cp $(CONTAINER_NAME):/ideas/outputs $(PWD)/outputs \
