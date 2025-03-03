# Login heroku
heroku container:login

# Création d'une image docker
docker build . -t api-cloud

# Start un container
docker run -p 5000:8000 -e PORT=5000 -v "$(pwd):/home/app" -it api-cloud