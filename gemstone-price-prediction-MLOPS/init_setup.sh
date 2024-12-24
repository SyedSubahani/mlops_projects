echo [$(date)]: "START"


echo [$(date)]: "creating env with python 3.12 version" 


conda create --prefix ./envmlops python=3.12 -y


echo [$(date)]: "activating the environment" 

source activate ./envmlops

echo [$(date)]: "installing the dev requirements" 

pip install -r requirements-dev.txt

echo [$(date)]: "END" 