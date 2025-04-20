# setup.sh
#!/bin/bash
python -m spacy download pt_core_news_sm
python -m nltk.download stopwords
python -m nltk.download punkt
