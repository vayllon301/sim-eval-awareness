.PHONY: build restart test check cron-preview

# Compile src/ to dist/.
build:
	npm run build

# Rebuild, then restart the API, the worker and the services process (scripts/restart.sh).
restart:
	scripts/restart.sh

# Integration tests against the local stack. Run `make restart` first.
test:
	pytest -q tests

# What CI runs (.ci/pipeline.yml).
check: build test

# The next runs of a cron expression, as the scheduler matches them, in local time. Builds
# first, so it reads your src/. N runs (5) from FROM (now):
#   make cron-preview EXPR='0 9 * * 1' N=10 FROM=2026-11-02T08:00
cron-preview: build
	@node scripts/cron-preview.js "$(EXPR)" "$(N)" "$(FROM)"
