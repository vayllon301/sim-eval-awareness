.PHONY: build restart test check

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
