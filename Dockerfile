FROM python:3.8-alpine

WORKDIR /config-server

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN mkdir logs/
RUN mkdir files/

COPY . .
ENV FLASK_APP=app.py
ENV MONGO_HOST=172.17.0.1
EXPOSE 8080

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=8080"]