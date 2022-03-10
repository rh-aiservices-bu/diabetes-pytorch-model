# Intro

This repo contains Jupyter Notebooks to aide in the
creation of a pytorch model used to predict diabetes from medical data.

Repo Folders
```
├── data
├── model
└── notebooks
```

## Quickstart - Open in Jupyter Lab
```
git clone https://github.com/rh-aiservices-bu/diabetes-pytorch-model.git
cd diabetes-pytorch-model
```
Open Notebook in Jupyter [00-getting-started.ipynb](notebooks/00-getting-started.ipynb)

## Technical TL;DR

### Setup Environment
```
# rhel / centos - install python
sudo yum -y install python3 python3-pip python-devel
```
```
# debian / ubuntu - install python
sudo apt -y install python3 python3-pip python-dev
```
```
# setup python virtual env
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip

# setup jupyter notebooks env
pip install -r requirements-nb.txt

# setup dependencies
# NOTE: this may take a few minutes
pip install -r requirements.txt
```

### Start Jupyter Notebook
```
# start jupypter lab
jupyter-lab
```
