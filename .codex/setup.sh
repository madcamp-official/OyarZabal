#!/usr/bin/env bash
set -euo pipefail

uv sync --extra dev
npm --prefix web ci
