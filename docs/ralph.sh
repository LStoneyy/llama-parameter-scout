#!/usr/bin/env bash
set -Eeuo pipefail

# Ralph loop for OpenCode with compact JSONL output.
# Place this file at: <repo>/docs/ralph.sh
#
# Usage:
#   ./docs/ralph.sh                  # up to 50 iterations
#   ./docs/ralph.sh 10               # up to 10 iterations
#   MAX_ITERATIONS=0 ./docs/ralph.sh # unlimited; completion marker still stops it
#
# Optional environment variables:
#   OPENCODE_BIN=opencode
#   OPENCODE_COMMAND=           # set to a custom command string to pass --command
#   OPENCODE_AUTO=1             # set to 0 to disable --auto
#   OPENCODE_MODEL=provider/model
#   OPENCODE_AGENT=build
#   OPENCODE_ATTACH=            # unsupported; must remain empty
#   OPENCODE_PRINT_LOGS=1       # write OpenCode's internal logs to the debug log
#   OPENCODE_LOG_LEVEL=INFO     # DEBUG, INFO, WARN, or ERROR
#   OPENCODE_PURE=0             # set to 1 to disable external plugins
#   RALPH_EXTERNAL_DIRS=/tmp/** # colon-separated external-directory patterns
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
OPENCODE_LOG_LEVEL="${OPENCODE_LOG_LEVEL:-INFO}"
OPENCODE_PURE="${OPENCODE_PURE:-0}"
RALPH_EXTERNAL_DIRS="${RALPH_EXTERNAL_DIRS:-/tmp/**}"
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

report() {
  printf '%s\n' "$*" | tee -a "$summary_log"
}

format_duration() {
  local total="$1"
  local hours=$((total / 3600))
  local minutes=$(((total % 3600) / 60))
  local seconds=$((total % 60))
  printf '%02d:%02d:%02d' "$hours" "$minutes" "$seconds"
}

command -v git >/dev/null 2>&1 || fail "git was not found in PATH"
command -v "$OPENCODE_BIN" >/dev/null 2>&1 || fail "$OPENCODE_BIN was not found in PATH"
command -v jq >/dev/null 2>&1 || fail "jq was not found in PATH"

if ! jq_help="$(jq --help 2>&1)"; then
  fail "jq --help failed"
fi
[[ "$jq_help" == *"--unbuffered"* ]] || fail "jq does not support --unbuffered"

if ! opencode_run_help="$("$OPENCODE_BIN" run --help 2>&1)"; then
  fail "$OPENCODE_BIN run --help failed"
fi
[[ "$opencode_run_help" == *"--format"* && "$opencode_run_help" == *"json"* ]] \
  || fail "$OPENCODE_BIN run does not support --format json"

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
[[ -z "$OPENCODE_ATTACH" ]] \
  || fail "OPENCODE_ATTACH is unsupported because Ralph must control server permissions"
[[ -z "${OPENCODE_PERMISSION:-}" ]] \
  || fail "OPENCODE_PERMISSION is managed by Ralph; unset it and use RALPH_EXTERNAL_DIRS"

REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)" \
  || fail "$SCRIPT_DIR is not inside a Git repository"
GIT_DIR="$(git -C "$REPO_ROOT" rev-parse --git-dir)"
if [[ "$GIT_DIR" != /* ]]; then
  GIT_DIR="${REPO_ROOT}/${GIT_DIR}"
fi

# Resolve the log directory relative to the repository so Git status checks can
# ignore generated Ralph logs without requiring a committed .gitignore.
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

IFS=':' read -r -a configured_external_dirs <<< "$RALPH_EXTERNAL_DIRS"
external_dirs=()
for pattern in "${configured_external_dirs[@]}"; do
  if [[ -n "$pattern" ]]; then
    [[ "$pattern" != "*" ]] || fail "RALPH_EXTERNAL_DIRS may not allow every external path"
    external_dirs+=("$pattern")
  fi
done
(( ${#external_dirs[@]} > 0 )) || fail "RALPH_EXTERNAL_DIRS must contain at least one pattern"

external_dirs+=(
  '~/.agents/skills/**'
  '~/.claude/skills/**'
  '~/.local/share/opencode/tool-output/**'
)
if [[ -n "${TMPDIR:-}" ]]; then
  external_dirs+=("${TMPDIR%/}/opencode/**")
fi

if ! external_dirs_json="$(
  printf '%s\n' "${external_dirs[@]}" \
    | jq -Rsc 'split("\n") | map(select(length > 0))'
)"; then
  fail "could not encode RALPH_EXTERNAL_DIRS"
fi

if ! OPENCODE_PERMISSION="$(
  jq -cn --argjson paths "$external_dirs_json" '
    {
      question: "deny",
      doom_loop: "deny",
      read: {
        "*.env": "deny",
        "*.env.*": "deny",
        "*.env.example": "allow"
      },
      external_directory: (
        reduce $paths[] as $path ({"*": "deny"}; .[$path] = "allow")
      )
    }
  '
)"; then
  fail "could not build OpenCode permission policy"
fi
export OPENCODE_PERMISSION

resolve_agents() {
  local primary="${OPENCODE_AGENT:-build}"
  local seen=""
  local candidate

  for candidate in "$primary" coder-cloud explorer-cloud reviewer-cloud; do
    case " $seen " in
      *" $candidate "*) ;;
      *)
        seen="$seen $candidate"
        printf '%s\n' "$candidate"
        ;;
    esac
  done
}

PERMISSION_ASK_FILTER='
  .permission as $rules
  | [
      $rules[]
      | select(.action == "ask")
      | {permission: .permission, pattern: .pattern}
    ]
  | unique_by([.permission, .pattern])
  | .[]
  | . as $ask
  | (
      [
        $rules[]
        | select(
            .permission == $ask.permission
            and .pattern == $ask.pattern
          )
      ]
      | last
      | .action
    ) as $last_action
  | select($last_action == "ask")
  | "\($ask.permission)=\($ask.pattern)"
'

while IFS= read -r agent; do
  debug_command=("$OPENCODE_BIN")
  if [[ "$OPENCODE_PURE" == "1" ]]; then
    debug_command+=(--pure)
  fi
  debug_command+=(debug agent "$agent")

  if ! agent_json="$("${debug_command[@]}" 2>/dev/null)"; then
    fail "could not resolve OpenCode agent: $agent"
  fi
  if ! printf '%s\n' "$agent_json" \
    | jq -e 'type == "object" and (.permission | type == "array")' >/dev/null 2>&1; then
    fail "invalid permission data for OpenCode agent: $agent"
  fi
  if ! unresolved_asks="$(printf '%s\n' "$agent_json" | jq -r "$PERMISSION_ASK_FILTER")"; then
    fail "could not evaluate permissions for OpenCode agent: $agent"
  fi
  if [[ -n "$unresolved_asks" ]]; then
    fail "agent $agent retains interactive permissions: ${unresolved_asks//$'\n'/, }"
  fi
done < <(resolve_agents)

JQ_FILTER='
def one_line:
  tostring
  | gsub("[\\r\\n\\t]+"; " ")
  | gsub("  +"; " ")
  | gsub("^ +| +$"; "");

def short:
  one_line
  | if length > 140 then .[0:137] + "..." else . end;

def error_text:
  if . == null then "unknown error"
  elif type == "string" then .
  elif type == "object" then (.data.message // .message // .name // tostring)
  else tostring
  end
  | short;

. as $event
| if $event.type == "tool_use" and $event.part.tool == "todowrite" then
    "TODO",
    (
      ($event.part.state.input.todos // [])[]
      | (
          if .status == "completed" then "[done]"
          elif .status == "in_progress" then "[work]"
          elif .status == "cancelled" then "[skip]"
          else "[    ]"
          end
        ) + " " + (.content // "untitled")
    )
  elif $event.type == "tool_use" and $event.part.tool == "task" then
    (($event.part.state.input.subagent_type // "agent") | short) as $agent
    | (($event.part.state.input.description // "task") | short) as $description
    | ($event.part.state.status // "completed") as $status
    | "TASK \($agent) \(if $status == "error" then "failed" else "done" end) - \($description)"
  elif $event.type == "tool_use" and $event.part.tool == "bash" then
    (($event.part.state.input.command // "unknown command") | short) as $command
    | ($event.part.state.status // "completed") as $status
    | (
        $event.part.state.metadata.exit
        // $event.part.state.metadata.exitCode
        // $event.part.state.metadata.exit_code
        // null
      ) as $exit_code
    | (
        "RUN \(if $status == "error" then "failed" else "done" end)"
        + (if $exit_code == null then "" else " (exit \($exit_code))" end)
        + " - \($command)"
      ),
      (if $status == "error" then "ERROR " + ($event.part.state.error | error_text) else empty end)
  elif $event.type == "tool_use" and ($event.part.tool == "edit" or $event.part.tool == "write") then
    "CHANGE " + (($event.part.state.input.filePath // "unknown file") | short)
  elif $event.type == "tool_use" and $event.part.tool == "apply_patch" then
    (
      [
        ($event.part.state.metadata.files // [])[]
        | (.relativePath // .movePath // .filePath // empty)
        | tostring
      ]
      | unique
      | join(", ")
    ) as $paths
    | "CHANGE " + (if $paths == "" then "apply_patch" else $paths end)
  elif $event.type == "text" and (($event.part.text // "") | length) > 0 then
    $event.part.text
  elif $event.type == "error" then
    "ERROR " + ($event.error | error_text)
  else
    empty
  end
'

iteration=1
while (( MAX_ITERATIONS == 0 || iteration <= MAX_ITERATIONS )); do
  timestamp="$(date -u '+%Y%m%dT%H%M%SZ')"
  log_prefix="${LOG_DIR}/${timestamp}-iteration-${iteration}"
  event_log="${log_prefix}.events.jsonl"
  debug_log="${log_prefix}.debug.log"
  summary_log="${log_prefix}.summary.log"
  iteration_started="$(date +%s)"
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

  command+=(run --format json --dir "$SCRIPT_DIR" --title "Ralph iteration ${iteration}")

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
  command+=("$prompt")

  if (( MAX_ITERATIONS == 0 )); then
    iteration_limit="unlimited"
  else
    iteration_limit="$MAX_ITERATIONS"
  fi

  {
    printf '\n=== Ralph iteration %d/%s ===\n' "$iteration" "$iteration_limit"
    printf 'Working directory: %s\n' "$SCRIPT_DIR"
    printf 'OpenCode version: '
    "$OPENCODE_BIN" --version 2>&1 || true
    printf 'Event log:   %s\n' "$event_log"
    printf 'Debug log:   %s\n' "$debug_log"
    printf 'Summary log: %s\n\n' "$summary_log"
  } | tee "$summary_log"

  set +e
  "${command[@]}" 2>"$debug_log" \
    | tee "$event_log" \
    | jq --unbuffered -r "$JQ_FILTER" \
    | tee -a "$summary_log"
  pipe_status=("${PIPESTATUS[@]}")
  set -e

  (( ${#pipe_status[@]} == 4 )) || fail "unexpected Ralph pipeline shape"
  opencode_status="${pipe_status[0]}"
  event_tee_status="${pipe_status[1]}"
  jq_status="${pipe_status[2]}"
  summary_tee_status="${pipe_status[3]}"

  (( event_tee_status == 0 )) \
    || fail "could not write event log; see $event_log"
  (( jq_status == 0 )) \
    || fail "could not parse OpenCode events; see $event_log and $debug_log"
  (( summary_tee_status == 0 )) \
    || fail "could not write summary log; see $summary_log"

  if grep -Fq 'message=asking' "$debug_log"; then
    fail "OpenCode requested interactive permission; adjust RALPH_EXTERNAL_DIRS. See $debug_log"
  fi

  if (( opencode_status != 0 )); then
    if grep -Fq 'Unexpected server error' "$event_log" \
      || grep -Fq 'Unexpected server error' "$debug_log"; then
      printf '%s\n' \
        "ralph: OpenCode returned a generic server error." \
        "ralph: Retry without a custom command:" \
        "ralph:   OPENCODE_COMMAND= ./docs/ralph.sh 1" \
        "ralph: Retry without external plugins:" \
        "ralph:   OPENCODE_COMMAND= OPENCODE_PURE=1 ./docs/ralph.sh 1" >&2
    fi
    fail "OpenCode exited with status $opencode_status; see $event_log and $debug_log"
  fi

  after_head="$(git -C "$REPO_ROOT" rev-parse --verify HEAD 2>/dev/null || printf 'UNBORN')"
  repository_dirty=0
  if [[ -n "$(repo_status)" ]]; then
    repository_dirty=1
  fi

  if (( repository_dirty == 1 )); then
    fail "OpenCode left uncommitted changes; see $event_log and $debug_log"
  fi

  completed=0
  if jq -e '
    select(
      .type == "text"
      and ((.part.text // "" | split("\n") | any(. == "RALPH_COMPLETE")))
    )
    | true
  ' "$event_log" >/dev/null 2>&1; then
    completed=1
  fi

  duration="$(format_duration "$(($(date +%s) - iteration_started))")"

  if (( completed == 1 )); then
    report ""
    if [[ "$after_head" != "$before_head" ]]; then
      report "Ralph completed after creating commit ${after_head}."
    else
      report "Ralph reports that all issues are already complete."
    fi
    report "Duration: ${duration}"
    exit 0
  fi

  if [[ "$after_head" == "$before_head" ]]; then
    fail "no new commit and no RALPH_COMPLETE marker; see $event_log and $debug_log"
  fi

  report ""
  report "Iteration ${iteration} committed successfully: ${after_head}"
  report "Duration: ${duration}"
  iteration=$((iteration + 1))
  sleep "$SLEEP_SECONDS"
done

fail "reached MAX_ITERATIONS=$MAX_ITERATIONS without RALPH_COMPLETE"
