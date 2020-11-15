NAME=project
VERSION=latest
IMG_W_TAG=${NAME}:${VERSION}
PORT=8888
P2P=${PORT}:${PORT}
VOLUME=${shell pwd}
V2V=${VOLUME}:/root/project
P2DOCKER=${VOLUME}/docker/Dockerfile

build:
		@docker build  -t ${IMG_W_TAG} -f ${P2DOCKER} .
run:
		@docker run -it --rm -v ${V2V} -p ${P2P} ${IMG_W_TAG}
