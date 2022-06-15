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
COPY requirements.txt .
# Install all requirements from requirements.txt
RUN pip install -r requirements.txt
RUN pip install gym
# Adding our python file to the root folder of the container
#COPY ./ ./minimal_rl_oaas/
# Run some commands after creation
#CMD ["python", "./environments/test.py"]