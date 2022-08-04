FROM python:3.8-slim-buster
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT ["streamlit","run"]
CMD ["app.py"]










# FROM ubuntu

# RUN apt-get update
# RUN apt-get install -y python3 python3-pip 
# RUN mkdir /opt/app

# WORKDIR /opt/app 

# COPY . .

# RUN pip3 install -r requirements.txt

# EXPOSE 8501

# ENTRYPOINT ["streamlit", "run"]
# CMD ["app.py"]