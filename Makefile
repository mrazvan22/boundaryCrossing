sshfs:
	sudo sshfs -o allow_other,defer_permissions,IdentityFile=/Users/razvan/.ssh/id_rsa razvan@sesame.csail.mit.edu:/research/boundaryCrossing /mnt/boundaryCrossing

unmount:
	sudo umount -f /mnt/boundaryCrossing

slurmCmd=cd /data/vision/polina/users/razvan/research/boundaryCrossing; srun -N 1 --ntasks-per-node=1 --gres=gpu:1  --pty zsh

launchServer:
	./setup_remote.sh 8877

gpuDev:
	ssh -t -X razvan@thyme.csail.mit.edu '$(slurmCmd)'


notebook_port=8822
remote_jupyter_dir=/data/vision/polina/users/razvan/research/boundaryCrossing

setup_remote:

	echo 'Loading zshrc'
	source /data/vision/polina/users/razvan/.zshrc
	#sleep 2
	source activate raz-vis2


	echo 'running jupyter notebook'
	nohup sudo jupyter-notebook --allow-root --no-browser --port=${notebook_port} --notebook-dir ${remote_jupyter_dir} --NotebookNotary.db_file=/tmp/ipython_hist.sqlite > nohup.out 2> nohup.err < /dev/null &


	jupyter-notebook list | grep ${notebook_port}
	
