docker build -t burrito .
docker run -d --name runner -it burrito sleep infinity
docker exec -it runner bash