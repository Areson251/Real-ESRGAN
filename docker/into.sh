#!/bin/bash

docker exec --user docker_esrgan -it ${USER}_esrgan \
    /bin/bash -c "cd /home/docker_esrgan; echo ${USER}_esrgan container; echo ; /bin/bash"