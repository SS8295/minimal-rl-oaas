# Dockerfile, Image, Container

# Fetch a docker image from DockerHub that has Ubuntu and install python 3.8
FROM ubuntu:20.04
RUN apt-get update
RUN apt-get install -y python3.8
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN apt-get install -y python3-pip

# Create a root folder
WORKDIR /Workspace
# Copy requirements in the current directory
#COPY requirements.txt .
# Install all requirements from requirements.txt
#RUN pip install -r requirements.txt
RUN pip install gym
RUN pip install matplotlib
RUN pip install pyyaml
RUN pip install pandas
RUN pip install torch
RUN pip install wandb
RUN pip install opencv-python
RUN apt-get update
#ENV DEBIAN_FRONTEND=noninteractive 
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
RUN apt-get install -y ffmpeg
RUN apt-get install libsm6 -y
RUN apt-get install libxext6 -y
RUN pip install networkx
RUN pip install dataframe-image
RUN pip install seaborn
#RUN pip install pyqt5
#RUN sudo apt-get install python-tk
# Adding our python file to the root folder of the container
#COPY ./ ./minimal_rl_oaas/
# Run some commands after creation
#CMD ["python", "./environments/test.py"]