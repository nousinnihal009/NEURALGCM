# Pre-start setup

Before running  `docker-compose up --build`  for the first time,
generate Flower authentication credentials:

    bash docker/make_htpasswd.sh <username> <password>

    # Example using a securely generated password:
    bash docker/make_htpasswd.sh flower_admin "$(openssl rand -hex 16)"

Store the generated password in your team's secrets manager.
The `docker/.htpasswd` file is gitignored and must not be committed.

If you skip this step, the nginx container will fail its healthcheck
and the api service will not start (depends_on: nginx with healthy).
