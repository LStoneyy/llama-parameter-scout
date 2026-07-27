#!/usr/bin/env bash
set -Eeuo pipefail

# Ralph loop for OpenCode.
# Place this file at: <repo>/docs/ralph.sh
#
# Usage:
#   ./docs/ralph.sh             # up to 50 iterations
#   ./docs/ralph.sh 10          # up to 10 iterations
#   MAX_ITERATIONS=0 ./docs/ralph.sh # unlimited; completion marker still stops it
#
# Optional environment variables:
#   OPENCODE_BIN=opencode
#   OPENCODE_COMMAND=           # set to a custom command string to pass --command
#   OPENCODE_AUTO=1             # set to 0 to disable --auto
#   OPENCODE_MODEL=provider/model
#   OPENCODE_AGENT=build
#   OPENCODE_ATTACH=http://localhost:4096
#   OPENCODE_PRINT_LOGS=1       # include OpenCode's internal logs
#   OPENCODE_LOG_LEVEL=DEBUG    # DEBUG, INFO, WARN, or ERROR
#   OPENCODE_PURE=0             # set to 1 to disable external plugins
#   SLEEP_SECONDS=2
#   ALLOW_DIRTY=0               # set to 1 only when intentional

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROMPT_FILE="${SCRIPT_DIR}/RALPH-PROMPT.md"
PRD_FILE="${SCRIPT_DIR}/PRD.md"
ISSUES_DIR="${SCRIPT_DIR}/issues"
LOG_DIR="${SCRIPT_DIR}/ralph-logs"

OPENCODE_BIN="${OPENCODE_BIN:-opencode}"
OPENCODE_COMMAND="${OPENCODE_COMMAND-}"
OPENCODE_AUTO="${OPENCODE_AUTO:-1}"
OPENCODE_MODEL="${OPENCODE_MODEL:-}"
OPENCODE_AGENT="${OPENCODE_AGENT:-}"
OPENCODE_ATTACH="${OPENCODE_ATTACH:-}"
OPENCODE_PRINT_LOGS="${OPENCODE_PRINT_LOGS:-1}"
OPENCODE_LOG_LEVEL="${OPENCODE_LOG_LEVEL:-DEBUG}"
OPENCODE_PURE="${OPENCODE_PURE:-0}"
SLEEP_SECONDS="${SLEEP_SECONDS:-2}"
ALLOW_DIRTY="${ALLOW_DIRTY:-0}"
MAX_ITERATIONS="${1:-${MAX_ITERATIONS:-50}}"

fail() {
  printf 'ralph: %s\n' "$*" >&2
  exit 1
}

is_boolean() {
  [[ "$1" == "0" || "$1" == "1" ]]
}

command -v git >/dev/null 2>&1 || fail "git was not found in PATH"
command -v "$OPENCODE_BIN" >/dev/null 2>&1 || fail "$OPENCODE_BIN was not found in PATH"

[[ -f "$PROMPT_FILE" ]] || fail "missing prompt file: $PROMPT_FILE"
[[ -f "$PRD_FILE" ]] || fail "missing PRD: $PRD_FILE"
[[ -d "$ISSUES_DIR" ]] || fail "missing issues directory: $ISSUES_DIR"
[[ "$MAX_ITERATIONS" =~ ^[0-9]+$ ]] || fail "MAX_ITERATIONS must be a non-negative integer"
[[ "$SLEEP_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail "SLEEP_SECONDS must be a non-negative number"
is_boolean "$OPENCODE_AUTO" || fail "OPENCODE_AUTO must be 0 or 1"
is_boolean "$OPENCODE_PRINT_LOGS" || fail "OPENCODE_PRINT_LOGS must be 0 or 1"
is_boolean "$OPENCODE_PURE" || fail "OPENCODE_PURE must be 0 or 1"
is_boolean "$ALLOW_DIRTY" || fail "ALLOW_DIRTY must be 0 or 1"
[[ "$OPENCODE_LOG_LEVEL" =~ ^(DEBUG|INFO|WARN|ERROR)$ ]] \
  || fail "OPENCODE_LOG_LEVEL must be DEBUG, INFO, WARN, or ERROR"

REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)" \
  || fail "$SCRIPT_DIR is not inside a Git repository"
GIT_DIR="$(git -C "$REPO_ROOT" rev-parse --git-dir)"
if [[ "$GIT_DIR" != /* ]]; then
  GIT_DIR="${REPO_ROOT}/${GIT_DIR}"
fi

# Resolve the log directory relative to the repository so that Git status checks
# can ignore generated Ralph logs without requiring a committed .gitignore.
SCRIPT_PREFIX="$(git -C "$SCRIPT_DIR" rev-parse --show-prefix)"
LOG_DIR_REL="${SCRIPT_PREFIX}ralph-logs"
LOCK_DIR="${GIT_DIR}/ralph-loop.lock"
mkdir -p "$LOG_DIR"

repo_status() {
  git -C "$REPO_ROOT" status --porcelain=v1 --untracked-files=all -- \
    . \
    ":(exclude)${LOG_DIR_REL}" \
    ":(exclude)${LOG_DIR_REL}/**"
}

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  fail "another Ralph loop appears to be running: $LOCK_DIR"
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

if [[ "$ALLOW_DIRTY" != "1" ]] && [[ -n "$(repo_status)" ]]; then
  fail "the repository is not clean; commit or stash changes first, or set ALLOW_DIRTY=1"
fi

cd "$SCRIPT_DIR"

iteration=1
while (( MAX_ITERATIONS == 0 || iteration <= MAX_ITERATIONS )); do
  timestamp="$(date -u '+%Y%m%dT%H%M%SZ')"
  log_file="${LOG_DIR}/${timestamp}-iteration-${iteration}.log"
  before_head="$(git -C "$REPO_ROOT" rev-parse --verify HEAD 2>/dev/null || printf 'UNBORN')"
  prompt="$(<"$PROMPT_FILE")"

  command=("$OPENCODE_BIN")
  if [[ "$OPENCODE_PRINT_LOGS" == "1" ]]; then
    command+=(--print-logs)
  fi
  command+=(--log-level "$OPENCODE_LOG_LEVEL")
  if [[ "$OPENCODE_PURE" == "1" ]]; then
    command+=(--pure)
  fi

  command+=(run --dir "$SCRIPT_DIR" --title "Ralph iteration ${iteration}")

  if [[ "$OPENCODE_AUTO" == "1" ]]; then
    command+=(--auto)
  fi
  if [[ -n "$OPENCODE_COMMAND" ]]; then
    command+=(--command "$OPENCODE_COMMAND")
  fi
  if [[ -n "$OPENCODE_MODEL" ]]; then
    command+=(--model "$OPENCODE_MODEL")
  fi
  if [[ -n "$OPENCODE_AGENT" ]]; then
    command+=(--agent "$OPENCODE_AGENT")
  fi
  if [[ -n "$OPENCODE_ATTACH" ]]; then
    command+=(--attach "$OPENCODE_ATTACH")
  fi
  command+=("$prompt")

  {
    printf '\n=== Ralph iteration %d ===\n' "$iteration"
    printf 'Working directory: %s\n' "$SCRIPT_DIR"
    printf 'Log: %s\n' "$log_file"
    printf 'OpenCode version: '
    "$OPENCODE_BIN" --version 2>&1 || true
    printf 'Command:'
    printf ' %q' "${command[@]}"
    printf '\n\n'
  } | tee "$log_file"

  set +e
  "${command[@]}" 2>&1 | tee -a "$log_file"
  opencode_status=${PIPESTATUS[0]}
  set -e

  if (( opencode_status != 0 )); then
    if grep -Fq 'Unexpected server error' "$log_file"; then
      cat >&2 <<EOF
ralph: OpenCode returned a generic server error.
ralph: This log includes --print-logs with log level ${OPENCODE_LOG_LEVEL}.
ralph: OpenCode also keeps native logs under ~/.local/share/opencode/log/ on macOS/Linux.
ralph: To disable any explicitly configured custom --command for isolation, run once with:
ralph:   OPENCODE_COMMAND= ./docs/ralph.sh 1
ralph: To additionally isolate external plugins, run once with:
ralph:   OPENCODE_COMMAND= OPENCODE_PURE=1 ./docs/ralph.sh 1
EOF
    fi
    fail "OpenCode exited with status $opencode_status; see $log_file"
  fi

  after_head="$(git -C "$REPO_ROOT" rev-parse --verify HEAD 2>/dev/null || printf 'UNBORN')"
  repository_dirty=0
  if [[ -n "$(repo_status)" ]]; then
    repository_dirty=1
  fi

  completed=0
  if grep -Fxq 'RALPH_COMPLETE' "$log_file"; then
    completed=1
  fi

  if (( repository_dirty == 1 )); then
    fail "OpenCode left uncommitted changes; stopping to avoid mixing issues. See $log_file"
  fi

  if (( completed == 1 )); then
    if [[ "$after_head" != "$before_head" ]]; then
      printf '\nRalph completed after creating commit %s.\n' "$after_head"
    else
      printf '\nRalph reports that all issues are already complete.\n'
    fi
    exit 0
  fi

  if [[ "$after_head" == "$before_head" ]]; then
    fail "no new commit and no RALPH_COMPLETE marker; stopping to prevent an endless loop. See $log_file"
  fi

  printf '\nIteration %d committed successfully: %s\n' "$iteration" "$after_head"
  iteration=$((iteration + 1))
  sleep "$SLEEP_SECONDS"
done

fail "reached MAX_ITERATIONS=$MAX_ITERATIONS without RALPH_COMPLETE"
