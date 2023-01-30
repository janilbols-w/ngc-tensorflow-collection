#!/bin/bash
# Utility to create list of python kernel_tests during build.

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
LIST_FILE="$THIS_DIR/tensorflow-source/$1/tests.list"
shift

set -e
if [[ $# -gt 0 ]]; then
    TEST="$1"
    TEST_SCRIPT="$THIS_DIR/tensorflow-source/${TEST#./}.py"
    if [[ ! -f "$TEST_SCRIPT" ]]; then
        # Try removing _gpu or _cpu suffix
        [[ -f "${TEST_SCRIPT%_[gc]pu.py}.py" ]] || exit 0
    fi
    shift
    echo ${TEST_SHARD_INDEX-NONE} ${TEST_TOTAL_SHARDS-NONE} $TEST_SCRIPT $@ \
        >> "$LIST_FILE"
fi
exit 0
