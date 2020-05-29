all:
	python train.py

convertPng:
	grep '^1' filenames.txt | xargs -I {} sh -c ' pngquant --quality=5-10 --output=/scratch/{}-fs8.png png_16bit/{}.png; convert -resize 50% /scratch/{}-fs8.png compressed_png/{}.png; rm /scratch/{}-fs8.png'


copy8192ToLocal:
	mkdir -p /home/razvan/xray_pngs_8192
	
	cat xray_pngs_8192.csv | xargs -I {} cp {} /home/razvan/xray_pngs_8192

launchJob:
	ZDOTDIR=/data/vision/polina/users/razvan srun --x11 -N 1 --ntasks-per-node=1 --cpus-per-task=40 --gres=gpu:8  --pty zsh

launchJobLight:
	ZDOTDIR=/data/vision/polina/users/razvan srun --x11 -N 1 --ntasks-per-node=1 --cpus-per-task=5 --gres=gpu:1  --pty zsh

linkZsh:
	 ln -s /data/vision/polina/users/razvan/.zshrc ~/.zshrc
	 ln -s /data/vision/polina/users/razvan/.zsh_aliases ~/.zsh_aliases
	 ln -s /data/vision/polina/users/razvan/.oh-my-zsh ~/.oh-my-zsh
	 ln -s /data/vision/polina/users/razvan/.vimrc ~/.vimrc
