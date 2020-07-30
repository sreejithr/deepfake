pip install -r requirements-latest.txt
git clone https://github.com/1adrianb/face-alignment
cd face-alignment
pip install -r requirements.txt
python setup.py install
cd ..
python core.py --config config/vox-256.yaml --checkpoint ~/Downloads/vox-cpk.pth.tar --relative --adapt_scale --source_image ~/Downloads/02.png --cpu
