NS ?= jihong
REPO = keras-gpu
VERSION ?= latest


DOCKER=nvidia-docker

VOLUMES = -v `pwd`:/root/workspace \
	-v ${HOME}/Datasets:/root/.keras/datasets
ENV = -e KERAS_BACKEND=tensorflow


.PHONY: bash

bash:
	$(DOCKER) run --rm -it $(PORTS) $(VOLUMES) $(ENV) $(NS)/$(REPO):$(VERSION) bash
