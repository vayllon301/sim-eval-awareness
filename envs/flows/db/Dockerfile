# The flows world's dev database (src/simtest/coding/machine.py, FLOWS): Postgres 15 with the
# app's schema and src/seed.ts already applied, so a box's database starts seeded and the same
# every time. The seed runs at build with the app's own tools (Prisma, ts-node), taken from
# LinuxArena's auto_workflow image without copying them in; seed.sh then dates the rows
# (dates.sql). The clock library lets the database read the box's clock (machine.py copies the
# setting in). Build from envs/:
#   docker build -t simtest-world-flows-db -f envs/flows/db/Dockerfile envs
ARG LA=linuxarena/env-auto_workflow:b7f3e189ebf436ccb783d1e6e12faf360baf5b9a
ARG PG=postgres:15.16
FROM ${LA} AS la

FROM ${PG} AS clock
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc libc6-dev libproc2-dev \
 && rm -rf /var/lib/apt/lists/*
COPY devbox/clock/clock.c /tmp/clock.c
RUN gcc -O2 -Wall -Wextra -Werror -shared -fPIC -o /libclock.so /tmp/clock.c

FROM ${PG}
COPY --from=clock /libclock.so /usr/local/lib/libclock.so
# Not the image's volume (/var/lib/postgresql/data), so the seeded data is part of the image.
ENV PGDATA=/var/lib/postgresql/flows
RUN --mount=type=tmpfs,target=/dev/shm \
    --mount=type=bind,from=la,target=/la \
    --mount=type=bind,source=coding/repos/flows_app,target=/src \
    --mount=type=bind,source=flows/db,target=/db \
    sh /db/seed.sh
USER postgres
CMD ["postgres"]
