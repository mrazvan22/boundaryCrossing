all:
	python train.py

convertPng:
	grep '^1' filenames.txt | xargs -I {} sh -c ' pngquant --quality=5-10 --output=/scratch/{}-fs8.png png_16bit/{}.png; convert -resize 50% /scratch/{}-fs8.png compressed_png/{}.png; rm /scratch/{}-fs8.png'


copy8192ToLocal:
	mkdir -p /home/razvan/xray_pngs_8192
	
	cat xray_pngs_8192.csv | xargs -I {} cp {} /home/razvan/xray_pngs_8192

launchJob:
	srun --x11 -N 1 --ntasks-per-node=1 --cpus-per-task=40 --gres=gpu:8  --pty zsh
