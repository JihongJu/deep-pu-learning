NS ?= jihong
REPO = keras-gpu
VERSION ?= latest


DOCKER=nvidia-docker

PORTS = -p 8888:8888
	VOLUMES = -v `pwd`:/root/workspace
	ENV = -e KERAS_BACKEND=tensorflow


.PHONY: bash

bash:
	$(DOCKER) run --rm -it $(PORTS) $(VOLUMES) $(ENV) $(NS)/$(REPO):$(VERSION) bash
