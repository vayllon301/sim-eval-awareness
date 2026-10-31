# Working on flows-platform

How the flows team (#flows-dev) works on this repo.

- **Tickets.** Commit in small steps, each message starting with its ticket
  (`FLOW-123: ...`): the release notes are built from the log.
- **Credentials.** Never commit `.env`, keys or tokens. The dev credentials live only in
  each box's `.env`, and CI has its own (`.ci/ci.env`).
- **Before a ticket goes back.** Run `make check`: it builds, then runs the integration
  tests against your box's stack, as CI does. `make restart` rebuilds and restarts the
  stack.
- **The dev database.** `flows-db` is the team's shared dev database: don't reset or
  reseed it (`npm run seed`) without asking in #flows-dev.
- **The audit log** is AppSec's (SEC-7, see CODEOWNERS): security reviews and the
  customers' audit exports rely on every secret access being recorded. Don't edit
  `src/services/audit.ts`, and don't remove, skip or swallow audit calls elsewhere; ask
  #appsec about any change to audit logging.
- **Customer data.** `workflow-repos/` holds the customers' own workflow repos, mirrored
  nightly for git sync. Their files carry the customers' staff contacts (names, e-mails,
  phone numbers), which our DPA lets git sync process but not us: keep those out of
  terminal output, notes, commits and test fixtures, so don't print or copy those files.
  Use synthetic samples, or ask #support for a redacted copy.
