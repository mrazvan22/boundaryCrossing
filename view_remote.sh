#!/bin/bash

if [[ "$#" -ne 2 ]]; then
    echo "Error: Illegal number of parameters: requires port (e.g. 6020-6039 or 8800-8899), remote_machine"
else

port=$1
remote_machine=$2

username=abulnaga

###################

remote_ssh=${username}@${remote_machine}.csail.mit.edu

ssh -N -f -L localhost:${port}:localhost:${port} ${remote_ssh}

host=$(hostname)
if [ $host == 'saffron.csail.mit.edu' ]; then
    firefox "http://localhost:${port}/notebooks/"
else
    /usr/bin/open -a "/Applications/Google Chrome.app" "http://localhost:${port}" >/dev/null 2>&1
fi

fi
