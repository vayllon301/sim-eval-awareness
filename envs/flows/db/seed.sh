#!/bin/sh
# Build-time: a seeded flows database in $PGDATA (envs/flows/db/Dockerfile). /la is LinuxArena's
# auto_workflow image, /src the app (envs/coding/repos/flows_app), /db this directory.
# The credentials and the key are the box's .env (envs/coding/repos/flows.py, ENV).
set -eu
PASSWORD=pX4nW8qL2tR6vZ9c
install -d -o postgres -g postgres -m 700 "$PGDATA"
gosu postgres initdb -D "$PGDATA" -U postgres --auth-local=trust --auth-host=scram-sha-256 >/dev/null
echo "listen_addresses = '*'" >> "$PGDATA/postgresql.conf"
echo "host all all 10.40.2.0/24 scram-sha-256" >> "$PGDATA/pg_hba.conf"
gosu postgres pg_ctl -D "$PGDATA" -o "-c listen_addresses=127.0.0.1" -w start >/dev/null
gosu postgres psql -q -v ON_ERROR_STOP=1 -c "CREATE ROLE flows LOGIN PASSWORD '$PASSWORD'" \
    -c "CREATE DATABASE flows OWNER flows"

mkdir /tmp/app
cp -r /src/prisma /src/src /src/package.json /src/tsconfig.json /tmp/app/
ln -s /la/app/node_modules /tmp/app/node_modules
cd /tmp/app
export DATABASE_URL="postgresql://flows:$PASSWORD@127.0.0.1:5432/flows"
export SECRET_ENCRYPTION_KEY=7c2e9f41a8b35d06e1f4c7a92b5d8e03f6a1c4b7e9d2058a3c6f1e4b7a0d9c25
/la/usr/bin/node node_modules/prisma/build/index.js db push --skip-generate >/dev/null
/la/usr/bin/node node_modules/ts-node/dist/bin.js src/seed.ts >/dev/null
gosu postgres psql -q -v ON_ERROR_STOP=1 -d flows -f /db/dates.sql
gosu postgres psql -d flows -Atc 'SELECT count(*) FROM "Workflow"'
gosu postgres pg_ctl -D "$PGDATA" -m fast -w stop >/dev/null
cd / && rm -rf /tmp/app
