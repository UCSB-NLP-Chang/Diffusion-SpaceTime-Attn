conda env create -f environment_replicate.yml
conda activate ldm
pip install bounding-box==0.1.3
pip install fairseq==0.12.2
pip install spacy==3.5.1
pip install nltk==3.8.1
pip install inflect==6.0.2
python -m spacy download en_core_web_sm
