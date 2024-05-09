You need the IP Address of your host system to start the docker-compose file for MLrun.
To get the IP address do the following:
- Open the Apple menu and click system settings.
- Click Network in the left panel.
- Select WiFi or Ethernet (if you are on a wired connection).
- Click Details next to the network you are using.
- Scroll down to find your IP Address.

You will also need a shared directory. The MLrn documentation has you creating this directory at ~/mlrun-data.

The commands below setup the environment variables needed by MLRun and start the docker compose file without
the Jupyter service. These should be put into a config.env file.

export HOST_IP=127.17.0.01
export SHARED_DIR=~/mlrun-data
mkdir $SHARED_DIR -p

docker-compose -f compose-with-jupyter-minio.yaml --env-file config.env up -d

docker-compose -f compose-with-jupyter.yaml --env-file config.env up -d

To clean up an existing shared directory.
rm -r ~/mlrun-data

The docker compose file without the Jupyter server will create the 3 services below:
- MLRun API: http://localhost:8080
- MLRun UI: http://localhost:8060
- Nuclio Dashboard/controller: http://localhost:8070

The docker compose file with the Jupyter server will also create a Jupyter service.
- Jupyter Lab: http://localhost:8888 

MLRun seems to be pretty pickly about the Python version. Use 3.9.1

pyenv install 3.9.13
pyenv global 3.9.13
pyenv versions

Install the MLRun library.
pip install mlrun

Use the MLRun SDK to point your development environment to the local MLrun instance that you just
installed.

mlrun.set_environment("http://localhost:8080", artifact_path="./")