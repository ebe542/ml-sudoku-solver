#!/usr/bin/env bash

# Run the complete verification suite at the end of a project milestone.
# Execute this file from Git Bash with: bash scripts/check_milestone.sh

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON="$PROJECT_ROOT/.venv/bin/python"
elif [[ -x "$PROJECT_ROOT/.venv/Scripts/python.exe" ]]; then
    PYTHON="$PROJECT_ROOT/.venv/Scripts/python.exe"
else
    echo "Error: No Python interpreter was found in $PROJECT_ROOT/.venv"
    echo "Create the local environment first: python -m venv .venv"
    exit 1
fi

cd "$PROJECT_ROOT" || exit 1

failed_checks=()
passed_checks=0
completed_checks=0

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    GREEN=$'\033[32m'
    RED=$'\033[31m'
    RESET=$'\033[0m'
else
    GREEN=""
    RED=""
    RESET=""
fi

MILESTONE_TMP_DIR="$(mktemp -d)"
trap 'rm -rf -- "$MILESTONE_TMP_DIR"' EXIT

run_check() {
    local name="$1"
    local log_file
    local remaining_checks
    shift

    ((completed_checks += 1))
    remaining_checks=$((total_checks - completed_checks))
    log_file="$MILESTONE_TMP_DIR/check-$completed_checks.log"

    printf '[%d/%d | remaining: %d] %s ... ' \
        "$completed_checks" "$total_checks" "$remaining_checks" "$name"

    if "$@" >"$log_file" 2>&1; then
        ((passed_checks += 1))
        printf '%sPASS%s\n' "$GREEN" "$RESET"
    else
        failed_checks+=("$name")
        printf '%sFAIL%s\n' "$RED" "$RESET"
        echo "--- Output from $name ---"
        sed 's/^/    /' "$log_file"
        echo "--- End of output ---"
    fi
}

excluded_scripts=(
    "scripts/train_model.py"
    "scripts/train_histogram_gradient_boosting.py"
)

is_excluded() {
    local script="$1"
    local excluded

    for excluded in "${excluded_scripts[@]}"; do
        if [[ "$script" == "$excluded" ]]; then
            return 0
        fi
    done

    return 1
}

mapfile -t python_scripts < <(find scripts -maxdepth 1 -type f -name '*.py' | sort)
scripts_to_run=()

for script in "${python_scripts[@]}"; do
    if ! is_excluded "$script"; then
        scripts_to_run+=("$script")
    fi
done

total_checks=$((2 + ${#scripts_to_run[@]}))

echo "ML Sudoku Solver milestone check"
echo "Project: $PROJECT_ROOT"
echo "Python:  $PYTHON"
echo "Checks:  $total_checks"
echo

run_check "Test suite" "$PYTHON" -m pytest -q
run_check "Script syntax" "$PYTHON" -m compileall -q scripts

for script in "${scripts_to_run[@]}"; do
    run_check "$script" "$PYTHON" "$script"
done

echo
echo "================================================================"
echo "Milestone check summary"
echo "================================================================"
echo "Passed: $passed_checks/$total_checks"

if ((${#failed_checks[@]} > 0)); then
    echo "Failed: ${#failed_checks[@]}"
    printf '  - %s\n' "${failed_checks[@]}"
    exit 1
fi

echo "All milestone checks passed."
echo
echo "Not executed automatically because they overwrite model files:"
printf '  - %s\n' "${excluded_scripts[@]}"
