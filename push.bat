@echo off

REM Login to Heroku
call heroku container:login

REM Create app on Heroku
call heroku create api-cloud-g1

REM Build Docker image
call docker build . -t api-cloud-g1

REM Tag Docker image for Heroku registry
call docker tag api-cloud-g1 registry.heroku.com/api-cloud-g1/web

REM Push Docker image to Heroku registry
call docker push registry.heroku.com/api-cloud-g1/web

REM Set the stack to container
call heroku stack:set container -a api-cloud-g1

REM Release the container on Heroku
call heroku container:release web -a api-cloud-g1

rem Ouverture de l'application
call heroku open -a api-cloud-g1

@echo Script terminé.
