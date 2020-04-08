sshfs:
	sudo sshfs -o allow_other,defer_permissions,IdentityFile=/Users/razvan/.ssh/id_rsa razvan@sesame.csail.mit.edu:/research/boundaryCrossing /mnt/boundaryCrossing

unmount:
	sudo umount -f /mnt/boundaryCrossing

slurmCmd=cd /data/vision/polina/users/razvan/research/boundaryCrossing; srun -N 1 --ntasks-per-node=1 --gres=gpu:1  --pty bash

launchServer:
	./setup_remote.sh 8877

gpuDev:
	ssh -t -X razvan@thyme.csail.mit.edu '$(slurmCmd)'
	