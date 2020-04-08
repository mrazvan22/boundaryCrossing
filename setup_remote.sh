#!/bin/bash

if [[ "$#" -ne 1 ]] || [[ "$1" -lt 8800 ]] || [[ "$1" -gt 8899 ]]; then
    echo "Error: Illegal number of parameters: requires notebook_port (e.g. 8800-8899)"
else

notebook_port=$1

###################

source /data/vision/polina/users/abulnaga/.bashrc
sleep 2
conda activate
source activate maz

###################

remote_jupyter_dir=/data/vision/polina/users/abulnaga/chest_xray/

nohup jupyter-notebook --no-browser --port=${notebook_port} --notebook-dir ${remote_jupyter_dir} --NotebookNotary.db_file=/tmp/ipython_hist.sqlite > nohup.out 2> nohup.err < /dev/null &

sleep 2

jupyter-notebook list | grep ${notebook_port}

fi
