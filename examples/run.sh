IMAGE_NAME=ghcr.io/tylera-nvidia/memorybenchmarks:cuda_12.5

USER_ID=$(id -u)
GROUP_ID=$(id -g)
CMDS="/bin/bash"

docker run  \
           -it \
           --gpus=all \
           --ipc=host \
           --privileged \
           --cap-add=SYS_ADMIN \
           -u $USER_ID:$GROUP_ID \
           -w /scratch \
           -v /tmp:/tmp \
           -v /home/$USER:/home/$USER \
           -v /home/scratch.tylera_sw/:/scratch \
           $IMAGE_NAME  /bin/bash


          #  -v /home/$USER:/home/matx \
#  $IMAGE_NAME fixuid /bin/bash
          
          