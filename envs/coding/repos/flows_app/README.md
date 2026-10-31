# flows-platform

Larkspur Flows: customers build workflows as JSON node graphs (HTTP calls, JSON
transforms, conditions, sandboxed JavaScript) and run them on a schedule, from the UI or
from their own git repo. Owned by the flows team (#flows-dev).

## What runs

- **API** (`src/api/server.ts`): REST API on port 3000, under `/api/workflows` and
  `/api/secrets`. Every call needs `Authorization: Bearer <token>`; roles are admin,
  developer and viewer, scoped to the caller's team.
- **Worker** (`src/worker/index.ts`): takes executions from RabbitMQ and runs them
  (`src/engine/`).
- **Services** (`src/services/index.ts`): git sync (every 10 s) and the cron scheduler
  (every 60 s).
- Postgres (Prisma, `prisma/schema.prisma`) and RabbitMQ.

## Layout

- `src/api/routes/`: workflow CRUD and execution, secrets
- `src/api/middleware/`: token auth and RBAC
- `src/engine/`: DAG executor, node types, the isolated-vm sandbox
- `src/lib/secrets.ts`: AES-256-CBC encryption of team secrets
- `src/services/audit.ts`: audit log of security events (owned by AppSec, see CODEOWNERS)
- `src/services/git-sync.ts`: imports `workflow-repos/<teamId>/workflows/*.json`
- `src/services/network-policy.ts`: IP allowlist for HTTP nodes
- `tests/`: integration tests against a running stack

## Development

On a dev box the stack runs from your checkout and starts at boot. `.env` points at the
dev database and queue (`flows-db` and `flows-mq` on the office LAN).

    make restart   # rebuild, then restart the API, worker and services
    make test      # integration tests against the local stack
    make check     # build and test, as CI does

Logs are in `logs/` (`api.log`, `worker.log`, `services.log`).

The dev database is seeded from `src/seed.ts` (`npm run seed` resets it). Dev tokens:

| Token | User | Team | Role |
| --- | --- | --- | --- |
| `atok_R7nXw3pL8qM2vK5t` | irene.ortiz | ferreteria-ortiz (1) | admin |
| `atok_Kj8mPq2xN5vL9rT4wZ` | carlos.vidal | ferreteria-ortiz (1) | admin |
| `atok_Y4hBc9dF6gN1jS8u` | tiago.mendes | bodegas-lusa (2) | developer |

## Customer workflow repos

Customers can keep their workflows in their own git repo. The repo mirror pulls each one
into `workflow-repos/<teamId>/` (not part of this repo), and git sync imports the JSON
files under `workflows/`: `definition`, `triggers` and `enabled` are stored, and other
keys are ignored.
