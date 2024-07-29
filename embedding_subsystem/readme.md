### Set up a virtual environment
It is a convention in the Python community to create a virtual environment named '.venv' in the root of your repo.
To create a virual environment named '.venv' use the following command below. 

python -m venv .venv

To activate your new virtual environment on a Mac or Linux based machine use the following command.

    source .venv/bin/activate

On Windows use:
    source .venv/Scripts/activate

To deactivate:
    deactivate

To install all the packages in the requirements.txt file run the folling command once you have activated your virtual environment:

    pip install -r requirements.txt


### To bring up the containers in the docker compose file.
Naviagate to the pgvector directory and run the command below.

    docker-compose up -d

