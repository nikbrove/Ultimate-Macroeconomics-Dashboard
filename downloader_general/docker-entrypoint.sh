#!/usr/bin/env sh
set -eu

# Always run main.py. The marker-file dance lives inside Python now so the
# LLM-role bootstrap (idempotent CREATE/ALTER + GRANTs) re-runs on every
# container start — required for password rotation, role provisioning on
# upgrades, and granting SELECT on tables added since the last bootstrap.
# Downloads themselves remain a one-shot.
exec python main.py
