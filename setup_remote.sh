#!/bin/zsh

if [[ "$#" -ne 1 ]] || [[ "$1" -lt 8800 ]] || [[ "$1" -gt 8899 ]]; then
    echo "Error: Illegal number of parameters: requires notebook_port (e.g. 8800-8899)"
else

notebook_port=$1

###################

echo 'Loading zshrc'
source /data/vision/polina/users/razvan/.zshrc
sleep 2
#conda activate
source activate raz-vis2

###################

remote_jupyter_dir=/data/vision/polina/users/razvan/research/boundaryCrossing

echo 'running jupyter notebook'
jupyter-notebook --no-browser --port=${notebook_port} --notebook-dir ${remote_jupyter_dir} --NotebookNotary.db_file=/tmp/ipython_hist.sqlite
#nohup jupyter-notebook --allow-root --no-browser --port=${notebook_port} --notebook-dir ${remote_jupyter_dir} --NotebookNotary.db_file=/tmp/ipython_hist.sqlite > nohup.out 2> nohup.err < /dev/null &

chmod -R 755 /home/razvan/.local

sleep 2

jupyter-notebook list


fi
