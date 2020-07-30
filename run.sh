pip install -r requirements-latest.txt
apt-get install libglib2.0-0
python core.py --config config/vox-256.yaml --checkpoint ~/Downloads/vox-cpk.pth.tar --relative --adapt_scale --source_image ~/Downloads/02.png --cpu
