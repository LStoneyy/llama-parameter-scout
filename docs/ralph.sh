#!/usr/bin/env bash
set -Eeuo pipefail

# Ralph loop for OpenCode.
# Place this file at: <repo>/docs/ralph.sh
#
# Usage:
#   ./ralph.sh                  # up to 50 iterations
#   ./ralph.sh 10               # up to 10 iterations
#   MAX_ITERATIONS=0 ./ralph.sh # unlimited; completion marker still stops it
#
# Optional environment variables:
#   OPENCODE_BIN=opencode
#   OPENCODE_COMMAND=implement  # set to an empty string to disable --command
#   OPENCODE_AUTO=1             # set to 0 to disable --auto
#   OPENCODE_MODEL=provider/model
#   OPENCODE_AGENT=build
#   OPENCODE_ATTACH=http://localhost:4096
#   SLEEP_SECONDS=2
#   ALLOW_DIRTY=0               # set to 1 only when intentional

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROMPT_FILE="${SCRIPT_DIR}/RALPH-PROMPT.md"
PRD_FILE="${SCRIPT_DIR}/PRD.md"
ISSUES_DIR="${SCRIPT_DIR}/issues"

OPENCODE_BIN="${OPENCODE_BIN:-opencode}"
OPENCODE_COMMAND="${OPENCODE_COMMAND-implement}"
OPENCODE_AUTO="${OPENCODE_AUTO:-1}"
OPENCODE_MODEL="${OPENCODE_MODEL:-}"
OPENCODE_AGENT="${OPENCODE_AGENT:-}"
OPENCODE_ATTACH="${OPENCODE_ATTACH:-}"
SLEEP_SECONDS="${SLEEP_SECONDS:-2}"
ALLOW_DIRTY="${ALLOW_DIRTY:-0}"
MAX_ITERATIONS="${1:-${MAX_ITERATIONS:-50}}"

fail() {
  printf 'ralph: %s\n' "$*" >&2
  exit 1
}

command -v git >/dev/null 2>&1 || fail "git was not found in PATH"
command -v "$OPENCODE_BIN" >/dev/null 2>&1 || fail "$OPENCODE_BIN was not found in PATH"

[[ -f "$PROMPT_FILE" ]] || fail "missing prompt file: $PROMPT_FILE"
[[ -f "$PRD_FILE" ]] || fail "missing PRD: $PRD_FILE"
[[ -d "$ISSUES_DIR" ]] || fail "missing issues directory: $ISSUES_DIR"
[[ "$MAX_ITERATIONS" =~ ^[0-9]+$ ]] || fail "MAX_ITERATIONS must be a non-negative integer"
[[ "$SLEEP_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail "SLEEP_SECONDS must be a non-negative number"
[[ "$OPENCODE_AUTO" == "0" || "$OPENCODE_AUTO" == "1" ]] || fail "OPENCODE_AUTO must be 0 or 1"
[[ "$ALLOW_DIRTY" == "0" || "$ALLOW_DIRTY" == "1" ]] || fail "ALLOW_DIRTY must be 0 or 1"

REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)" \
  || fail "$SCRIPT_DIR is not inside a Git repository"
GIT_DIR="$(git -C "$REPO_ROOT" rev-parse --git-dir)"
if [[ "$GIT_DIR" != /* ]]; then
  GIT_DIR="${REPO_ROOT}/${GIT_DIR}"
fi

LOG_DIR="${GIT_DIR}/ralph-logs"
LOCK_DIR="${GIT_DIR}/ralph-loop.lock"
mkdir -p "$LOG_DIR"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  fail "another Ralph loop appears to be running: $LOCK_DIR"
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

if [[ "$ALLOW_DIRTY" != "1" ]] && [[ -n "$(git -C "$REPO_ROOT" status --porcelain=v1)" ]]; then
  fail "the repository is not clean; commit or stash changes first, or set ALLOW_DIRTY=1"
fi

cd "$SCRIPT_DIR"

iteration=1
while (( MAX_ITERATIONS == 0 || iteration <= MAX_ITERATIONS )); do
  timestamp="$(date -u '+%Y%m%dT%H%M%SZ')"
  log_file="${LOG_DIR}/${timestamp}-iteration-${iteration}.log"
  before_head="$(git -C "$REPO_ROOT" rev-parse --verify HEAD 2>/dev/null || printf 'UNBORN')"
  prompt="$(<"$PROMPT_FILE")"

  printf '\n=== Ralph iteration %d ===\n' "$iteration"
  printf 'Working directory: %s\n' "$SCRIPT_DIR"
  printf 'Log: %s\n\n' "$log_file"

  command=("$OPENCODE_BIN" run --dir "$SCRIPT_DIR" --title "Ralph iteration ${iteration}")

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

  set +e
  "${command[@]}" 2>&1 | tee "$log_file"
  opencode_status=${PIPESTATUS[0]}
  set -e

  if (( opencode_status != 0 )); then
    fail "OpenCode exited with status $opencode_status; see $log_file"
  fi

  after_head="$(git -C "$REPO_ROOT" rev-parse --verify HEAD 2>/dev/null || printf 'UNBORN')"
  repository_dirty=0
  if [[ -n "$(git -C "$REPO_ROOT" status --porcelain=v1)" ]]; then
    repository_dirty=1
  fi

  completed=0
  if grep -Fq 'RALPH_COMPLETE' "$log_file"; then
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
