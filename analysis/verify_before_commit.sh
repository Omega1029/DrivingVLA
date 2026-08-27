#!/bin/bash
# Verification gate for the autonomous loop (see AUTOLOOP.md). Must exit 0 before any commit.
# Deliberately conservative: a false "fail" costs one wasted cycle; a false "pass" risks
# committing a broken paper or a bad number. Extend this as new checks become relevant.
set -u
cd "$(dirname "$0")/.."
FAIL=0

echo "=== 1. LaTeX structural balance (changed .tex files) ==="
CHANGED_TEX=$(git diff --cached --name-only --diff-filter=ACM | grep '\.tex$' || true)
if [ -z "$CHANGED_TEX" ]; then
  echo "  no .tex files staged, skipping"
else
  for f in $CHANGED_TEX; do
    python3 - "$f" <<'PYEOF'
import sys, re
f = sys.argv[1]
s = open(f).read()
bad = False
for env in ('abstract','table','table*','tabular','itemize','enumerate','IEEEkeywords',
            'document','figure','equation','align'):
    b = s.count('\\begin{%s}' % env)
    e = s.count('\\end{%s}' % env)
    if b != e:
        print(f"  FAIL {f}: \\begin{{{env}}}={b} \\end{{{env}}}={e}")
        bad = True
labels = set(re.findall(r'\\label\{([^}]+)\}', s))
refs = set(re.findall(r'\\ref\{([^}]+)\}', s))
dangling = refs - labels
if dangling:
    print(f"  FAIL {f}: dangling \\ref with no \\label: {dangling}")
    bad = True
if not bad:
    print(f"  OK {f}")
sys.exit(1 if bad else 0)
PYEOF
    [ $? -ne 0 ] && FAIL=1
  done
fi

echo "=== 2. Closed-loop output aggregation (if output dirs changed) ==="
if git diff --cached --name-only --diff-filter=ACM | grep -q '^analysis/RESULTS_clean_harness'; then
  if python3 analysis/compare_clean_vs_archived.py > /tmp/gate_agg_check.txt 2>&1; then
    echo "  OK aggregation script runs clean"
  else
    echo "  FAIL aggregation script errored:"; tail -10 /tmp/gate_agg_check.txt; FAIL=1
  fi
fi

echo "=== 3. No paper .tex content changes without a NEW PROGRESS.md entry ==="
if [ -n "$CHANGED_TEX" ]; then
  # Require actual added lines in PROGRESS.md's staged diff, not just presence in the changed
  # list -- a file staged for an unrelated reason earlier in the same index would otherwise
  # satisfy a weaker check without a real new entry existing.
  ADDED=$(git diff --cached -- PROGRESS.md | grep -c '^+[^+]' || true)
  if [ "$ADDED" -lt 1 ]; then
    echo "  FAIL: .tex changed but PROGRESS.md has no new added lines staged (AUTOLOOP.md rule 1)"
    FAIL=1
  else
    echo "  OK PROGRESS.md has $ADDED new line(s) staged alongside the .tex change"
  fi
fi

echo "=== 4. No force-push flags anywhere in staged shell scripts ==="
CHANGED_SH=$(git diff --cached --name-only --diff-filter=ACM | grep '\.sh$' | grep -v 'verify_before_commit\.sh$' || true)
if [ -n "$CHANGED_SH" ] && git diff --cached -- $CHANGED_SH | grep -v '^+.*grep' | grep -qE '(push --force|push -f\b|reset --hard.*origin)'; then
  echo "  FAIL: a staged script contains a force-push or hard-reset-to-origin pattern"
  FAIL=1
else
  echo "  OK"
fi

if [ $FAIL -ne 0 ]; then
  echo; echo "GATE FAILED. Do not commit. Fix or log a flagged entry in PROGRESS.md instead."
  exit 1
fi
echo; echo "GATE PASSED."
exit 0
