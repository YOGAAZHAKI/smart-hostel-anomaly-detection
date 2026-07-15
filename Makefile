.PHONY: install train test run

install:
	pip install -r requirements.txt

train:
	set PYTHONIOENCODING=utf-8 && python train_classifier.py --mode train --model vit --epochs 20

test:
	set PYTHONIOENCODING=utf-8 && python test_model.py

run: test
