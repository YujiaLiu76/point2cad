#!/usr/bin/env bash
if [ ! -z "${DEBUG}" ]; then
    set -x
fi

USER_ID=$(stat -c %u /work/point2cad)
GROUP_ID=$(stat -c %g /work/point2cad)

groupadd -g $GROUP_ID usergroup
useradd -m -l -u $USER_ID -g usergroup user

if [ ! -z "${DEBUG}" ]; then
    env
    whoami
    groups
    python -c "import torch; print(torch.cuda.is_available())"
    ls -l /dev/nvidia*
    gosu user env
    gosu user whoami
    gosu user groups
    gosu user python -c "import torch; print(torch.cuda.is_available())"
    gosu user ls -l /dev/nvidia*
fi

exec gosu user "$@"
