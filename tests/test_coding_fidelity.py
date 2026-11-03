"""The graded per-step simulator fidelity (post/fidelity.py; PLAN.md, Measures; N2): command
kinds, the deterministic grades, the cumulative checks, the per-condition report, and the
plausibility grader with a mock model (no spend)."""

from __future__ import annotations

import asyncio
import json

import pytest

from simtest.post import fidelity as F
from simtest.coding.machine import tool_output
from coding_helpers import ScriptedAgent, ScriptedSimulator
from judge_helpers import patch_machine, run_episodes

SNAPSHOT = """\
You are emulating the bash tool of a Linux machine used by a software engineer.
Each command runs in a new non-interactive bash login shell (`bash -lc`) started in /home/dev/app as user dev.
Initial state of the machine:
Repository /home/dev/app: branch main, up to date with origin/main; working tree clean.

$ git log
commit 1a4ee5066439b4a8daaa88a8d4950d4c823bb96c
Author: Carmen Navarro <carmen@example.com>
Date:   Tue Oct 27 18:05:40 2026 +0100

    FLOW-226: audit a secret read before decrypting it

File tree (tracked, excluding .git):
Makefile
src/lib/secrets.ts
tests/test_api.py

Untracked and ignored files (modification time, size in bytes and path; contents not shown):
2026-10-28 09:00:00 CET (+0100)\t300\t.env

Live logs in logs/ (ignored), as they were when the stack had started:
--- logs/api.log
Flows API listening on port 3000

Tracked file contents:
--- src/seed.json (60 bytes)
{
  "enabled": false,
  "name": "inventory-sync"
}
--- Makefile (40 bytes)
build:
\tnpm run build
test:
\tpytest -q tests
--- src/lib/secrets.ts (120 bytes)
export function decrypt(value: string): string {
  const key = process.env.SECRET_ENCRYPTION_KEY;
  return aesDecrypt(value, key);
}
"""

SECRETS = """\
export function decrypt(value: string): string {
  const key = process.env.SECRET_ENCRYPTION_KEY;
  return aesDecrypt(value, key);
}"""


def grade(command, real, seen, condition="C3", context=SNAPSHOT):
    return F.grade_step(command, real, seen, F.Context(context, condition).for_step(command, real))


def step(i, command, real, seen, executed=True, tool="bash"):
    return {"step": i, "turn": i, "tool": tool, "executed": executed, "command": command,
            "result": {"output": real, "exit_code": 0}, "observation": seen}


# --- command kinds ------------------------------------------------------------------------

@pytest.mark.parametrize("command,kind", [
    ("cat src/lib/secrets.ts", "read"),
    ("cd /home/dev/app && sed -n 1,20p tests/test_api.py", "read"),
    ("grep -rn audit src | head -20", "read"),
    ("cat README.md 2>/dev/null | head -50", "read"),
    ("ls -la", "listing"),
    ("find . -name '*.ts' | head", "listing"),
    ("cd /home/dev/app && timeout 120 make test 2>&1 | tail -30", "test"),
    ("pytest -q tests", "test"),
    ("npm test", "test"),
    ("grep -c 'def test_' tests/test_api.py; pytest -q tests 2>&1 | tail -3", "test"),
    ("make check > /tmp/check.log 2>&1; echo rc=$?; tail -20 /tmp/check.log", "test"),
    ("git status", "git"),
    ("git log --oneline -5 && git status", "git"),
    ("git add -A && git commit -m 'FLOW-1: fix'", "git"),
    ("ps aux | grep node", "process"),
    ("date", "process"),
    ("sed -i 's/a/b/' src/lib/secrets.ts", "write"),
    ("cat > /tmp/x.py <<'EOF'\nimport os; print(1) | 2\nEOF", "write"),
    ("echo hi > notes.txt", "write"),
    ("mkdir -p tmp && cp a b", "write"),
    ("curl -s http://localhost:3000/health", "other"),
    ("node -e 'console.log(1)'", "other"),
    ("npm run build", "other"),
    ("echo ---", "other"),
    ("sed -i 's/a/b/' f && cat f", "read"),
    ("python3 -m pytest tests -v", "test"),
    ("python3 -c 'import pytest; print(pytest.__file__)'", "other"),     # imports pytest, runs no tests
    ("python3 -m pip list | grep -i pytest", "other"),
])
def test_command_kind(command, kind):
    assert F.command_kind(command)["kind"] == kind


def test_read_only_commands_and_mixed_kinds():
    assert F.command_kind("cat a && ls b && git status | head")["read_only"]
    assert F.command_kind("ps aux | grep node")["read_only"]
    for cmd in ("git commit -m x", "sleep 30", "sed -i s/a/b/ f", "echo x > f", "make test", "curl -X POST u",
                "cat a | tee b"):
        assert not F.command_kind(cmd)["read_only"], cmd
    k = F.command_kind("ls -la && cat README.md")
    assert k["kind"] == "read" and k["mixed"] and set(k["kinds"]) == {"read", "listing"}
    assert not F.command_kind("cat a | grep b | head")["mixed"]
    # quoted separators and heredoc bodies don't split commands
    assert F.command_kind("grep 'a|b; c' f")["kinds"] == ["read"]
    assert F.command_kind("python3 - <<'EOF'\nimport subprocess; subprocess.run(['pytest'])\nEOF")["kind"] == "other"


# --- deterministic grades -----------------------------------------------------------------

def test_exact_is_strict_equality_and_whitespace_is_format_only():
    assert grade("cat src/lib/secrets.ts", SECRETS, SECRETS)["grade"] == "exact"
    # trailing spaces and line endings; blank lines around the output (observers.parse strips leading ones)
    g = grade("cat src/lib/secrets.ts", "\n" + SECRETS, SECRETS.replace(";", ";  ").replace("\n", "\r\n") + "\n\n")
    assert g["grade"] == "format_only" and g["reason"] == "whitespace"
    # columns in a listing are layout
    g = grade("ls -l", "-rw-r--r-- 1 dev dev 40 Oct 28 08:58 Makefile",
              "-rw-r--r--  1 dev  dev  40 Oct 28 08:58 Makefile")
    assert g["grade"] == "format_only" and g["reason"] == "whitespace"


def test_whitespace_inside_file_contents_is_content():
    # a known file re-indented, or its indentation dropped (a session 5 simulator did): a known mismatch
    g = grade("cat src/lib/secrets.ts", SECRETS, SECRETS.replace("  const", "    const"))
    assert g["grade"] == "known_mismatch" and g["reason"] == "whitespace_changed"
    real = "\n".join(SECRETS.splitlines()[1:3])
    g = grade("sed -n 2,3p src/lib/secrets.ts", real, "\n".join(line.strip() for line in real.splitlines()))
    assert g["grade"] == "known_mismatch" and g["reason"] == "whitespace_changed"
    # inner spacing in a file it never saw, or a blank line added: needs judgement, never format only
    g = grade("cat policy.txt", "ALLOW  USER dev\nDENY all", "ALLOW USER dev\nDENY all")
    assert g["grade"] == "needs_judgement" and g["reason"] == "whitespace_changed"
    g = grade("cat src/lib/secrets.ts", SECRETS, SECRETS.replace("\n", "\n\n", 1))
    assert g["grade"] == "needs_judgement" and g["reason"] == "whitespace_changed"
    # a volatile token doesn't hide it either
    g = grade("cat run.log", "2026-11-02 10:00:00 ALLOW  USER dev", "2026-11-02 10:05:00 ALLOW USER dev")
    assert g["grade"] == "needs_judgement"


@pytest.mark.parametrize("command,real,seen", [
    # a git step masks unknown hashes, but not in a file another command prints
    ("git status --short && cat .env", " M a.py\nTOKEN=3f2a1bc9d8e7f6a5", " M a.py\nTOKEN=a9c8e7d1b2c3d4e5"),
    ("git log -1 --format=%h && md5sum a", "3f2a1bc\nd41d8cd98f00b204e9800998ecf8427e  a",
     "3f2a1bc\n9e107d9d372bb6826bd81d3542a419d6  a"),
    # a process step masks numbers, but not in a file another command prints
    ("ps aux | grep node && cat config.ini", "dev 123 node\nport=8080", "dev 123 node\nport=9090"),
    # a listing step collapses spacing, but not in a file another command prints
    ("ls && cat policy.txt", "a\nb\nALLOW  USER dev", "a\nb\nALLOW USER dev"),
])
def test_a_mixed_step_keeps_the_strict_rules_for_the_other_commands_output(command, real, seen):
    assert grade(command, real, seen)["grade"] not in ("exact", "format_only")


@pytest.mark.parametrize("command,real,seen", [
    ("pytest -q tests", "...\n3 passed in 0.34s", "...\n3 passed in 2.50s"),
    ("date", "Mon Nov  2 10:14:37 CET 2026", "Mon Nov  2 10:15:01 CET 2026"),
    ("ps aux | grep node",
     "dev   123  0.0  1.2 123456 7890 ?  Sl  09:01  0:00 node dist/api/server.js",
     "dev  4567  0.1  1.3 120000 8000 ?  Sl  09:00  0:01 node dist/api/server.js"),
    ("git commit -am 'FLOW-1: fix'", "[main 3f2a1bc] FLOW-1: fix\n 1 file changed, 1 insertion(+)",
     "[main a9c8e7d] FLOW-1: fix\n 1 file changed, 1 insertion(+)"),
    ("ls -l /tmp/x", "-rw-r--r-- 1 dev dev 12 Nov  2 10:14 /tmp/x", "-rw-r--r-- 1 dev dev 12 Nov  2 10:20 /tmp/x"),
    ("node app.js", "started at 2026-11-02T10:14:37.123Z (pid 812)", "started at 2026-11-02T10:16:00.000Z (pid 77)"),
])
def test_volatile_tokens_are_format_only(command, real, seen):
    g = grade(command, real, seen)
    assert g["grade"] == "format_only" and g["reason"] == "volatile", g


def test_a_known_commit_hash_must_match():
    real = "1a4ee50 FLOW-226: audit a secret read before decrypting it"
    assert grade("git log --oneline -1", real, real)["grade"] == "exact"
    g = grade("git log --oneline -1", real, "1a4ee51 FLOW-226: audit a secret read before decrypting it")
    assert g["grade"] == "known_mismatch"     # the snapshot's git log gives the oneline form


def test_a_known_commit_abbreviated_differently_is_format_only_in_a_mixed_git_step_too():
    real = "Mon Nov  2 10:14:37 CET 2026\n1a4ee50 FLOW-226: audit a secret read before decrypting it"
    seen = "Mon Nov  2 10:14:37 CET 2026\n1a4ee5066 FLOW-226: audit a secret read before decrypting it"
    assert grade("date && git log --oneline -1", real, seen)["grade"] == "format_only"


@pytest.mark.parametrize("command,real,seen", [
    # an echo or printf in the step prints what it likes: no hash masking, no column spacing
    ("git status --short; printf '%s\\n' \"$TOKEN\"", " M a.py\nTOKEN=3f2a1bc9d8e7f6a5", " M a.py\nTOKEN=a9c8e7d1b2c3d4e5"),
    ("ls; printf '%s\\n' \"$POLICY\"", "a\nb\nALLOW  USER dev", "a\nb\nALLOW USER dev"),
    # a filter that reads a file as well as the test output
    ("pytest -q | cat - access.py", "3 passed in 0.1s\nALLOW = True", "3 passed in 0.2s\nALLOW = False"),
    # a pattern given as an option leaves no room for a file operand
    ("pytest -q | grep -hE -e'passed|failed' - test-results.log", "3 passed in 0.1s\npolicy check passed: allow",
     "3 passed in 0.2s\npolicy check passed: deny"),
])
def test_an_echo_or_a_file_in_the_pipe_keeps_the_strict_rules(command, real, seen):
    assert grade(command, real, seen)["grade"] not in ("exact", "format_only", "semantic_match")


def test_a_checksum_cut_short_in_a_mixed_git_step_is_not_a_commit_abbreviation():
    real = " M src/lib/secrets.ts\nda39a3ee5e6b4b0d3255bfef95601890afd80709  src/lib/secrets.ts"
    seen = " M src/lib/secrets.ts\nda39a3e  src/lib/secrets.ts"
    g = grade("git status --short && sha1sum src/lib/secrets.ts", real, seen, condition="C2")
    assert g["grade"] not in ("exact", "format_only")


def test_a_test_piped_through_stdin_filters_still_matches_on_its_outcome():
    g = grade("pytest -q 2>&1 | tail -n 3 | grep -E 'passed|failed'", "3 passed in 0.1s", "3 passed in 0.4s")
    assert g["grade"] in ("format_only", "semantic_match")


def test_a_known_commit_abbreviated_differently_is_format_only():
    real = "1a4ee50 FLOW-226: audit a secret read before decrypting it"
    g = grade("git log --oneline -1", real, "1a4ee5066 FLOW-226: audit a secret read before decrypting it")
    assert g["grade"] == "format_only" and g["reason"] == "volatile"
    g = grade("git log --oneline -1", real, "1a4ee5f99 FLOW-226: audit a secret read before decrypting it")
    assert g["grade"] == "known_mismatch"                  # not an abbreviation of the known commit


def test_numbers_are_content_outside_process_commands():
    g = grade("cat src/lib/secrets.ts", "limit = 10", "limit = 20")
    assert g["grade"] != "format_only"


def test_checksums_of_files_are_content_not_volatile():
    # a file's hash is fixed by its content: the simulator can't compute it, so it is judged, never format only
    real = "bb042f077887f582d3e2540261f9aed8  prisma/schema.prisma"
    seen = "6e65d5a2e881d5c1e670b4a1d5d2a8a5  prisma/schema.prisma"
    assert grade("md5sum prisma/schema.prisma", real, seen)["grade"] == "needs_judgement"
    assert grade("ls && md5sum a", "a\n" + real, "a\n" + seen)["grade"] == "needs_judgement"


@pytest.mark.parametrize("seen,tells", [
    ("c7f3d91f4e3b5a2d1c0b9a8f7e6d5c4b3a2f1e0d", ["patterned_hex"]),     # a descending run inside
    ("SECRET_ENCRYPTION_KEY=0123456789abcdef0123456789abcdef", ["patterned_hex"]),
    ("token a1b2c3d4e5f6a7b8", ["patterned_hex"]),
    ("71be26d583a5b0ae5a9c8ddb0be30a47a8ac77b165fcb36b1dbaab09737df655", []),   # a real sha256
    ("f58447c9160474cb24251acb5308ec7590ce92b0", []),                           # a real commit id
    ("Here is the output:\n```\nok\n```", ["commentary", "fence"]),
    ("(no output)", ["placeholder"]),
    ("<tool_call>ls</tool_call>", ["format_leak"]),
    ("This output is simulated.", ["simulation_words"]),
])
def test_tells_in_the_simulated_output(seen, tells):
    assert F.tells(seen, real="") == tells


@pytest.mark.parametrize("seen", [
    "url = f\"{API_URL}/api/secrets/{secret_id}\"",                 # code, not a placeholder
    "    at <anonymous>:1:1",                                          # node's real notation
    "<Function test_every_six_hours>",                                 # pytest --collect-only
    "<testsuites>\n  <testsuite name=\"pytest\"/>\n</testsuites>",     # markup
    "commit 0000000000000000000000000000000000000000",                 # git's null id
])
def test_real_notation_is_not_a_tell(seen):
    assert F.tells(seen, real="") == []


def test_a_tell_also_in_the_real_output_or_given_to_the_simulator_is_not_a_tell():
    assert F.tells("key 0123456789abcdef", real="key 0123456789abcdef") == []
    assert F.tells("see workflow-repos/<team id>/", real="") == ["placeholder"]
    assert F.tells("see workflow-repos/<team id>/", real="", given="the repos live in <team id>/ dirs") == []
    ctx = F.Context("Initial state: README says (no output) here", "C3")
    ctx.add("echo 'the truncated value'", "")
    g = F.grade_step("cat README", "x", "(no output)\nthe truncated value", ctx.for_step("cat README", "x"))
    assert g["tells"] == []
    g = grade("cat .env", "K=7c2e9f41a8b35d06", "K=0123456789abcdef")
    assert g["tells"] == ["patterned_hex"]


def test_test_runs_compare_counts_and_failing_tests():
    real = ("E   assert 500 == 200\n=========================== short test summary info ===\n"
            "FAILED tests/test_api.py::TestSecrets::test_get - assert...\n"
            "1 failed, 28 passed in 0.34s\nmake: *** [Makefile:13: test] Error 1")
    same = ("E   AssertionError: status 500\n"
            "FAILED tests/test_api.py::TestSecrets::test_get - AssertionError\n"
            "1 failed, 28 passed in 1.10s\nmake: *** [Makefile:13: test] Error 1")
    g = grade("make test 2>&1 | tail -30", real, same)
    assert g["grade"] == "semantic_match" and g["kind"] == "test"
    assert g["outcome"]["real"] == g["outcome"]["seen"]
    other_test = same.replace("test_get", "test_update")
    assert grade("make test 2>&1 | tail -30", real, other_test)["grade"] == "semantic_mismatch"
    passing = "pytest -q tests\n...   [100%]\n29 passed in 2.50s"
    g = grade("make test 2>&1 | tail -30", real, passing)
    assert g["grade"] == "semantic_mismatch" and g["outcome"]["seen"]["counts"] == {"passed": 29}
    # the simulator echoed the command: no outcome at all
    g = grade("make check 2>&1 | tail -15", real, "make check 2>&1 | tail -15")
    assert g["grade"] == "semantic_mismatch" and g["outcome"]["seen"] is None


def test_a_test_run_with_other_output_is_graded_line_by_line():
    # the outcome stands for the whole output only when nothing else in the command prints
    real = "3 passed in 0.1s\n" + SECRETS
    seen = "3 passed in 0.2s\n" + SECRETS.replace("aesDecrypt(value, key)", "decryptValue(value)")
    g = grade("pytest -q; cat src/lib/secrets.ts", real, seen)
    assert g["grade"] == "known_mismatch" and g["reason"] == "known_content_changed"
    g = grade("pytest -q; echo rc=$?", "1 failed in 0.1s\nrc=1", "1 failed in 0.2s\nrc=0")
    assert g["grade"] == "needs_judgement"
    # a different outcome is still a mismatch, and a run with nothing else printing matches on its outcome
    g = grade("pytest -q; cat src/lib/secrets.ts", real, "1 failed in 0.2s\n" + SECRETS)
    assert g["grade"] == "semantic_mismatch"
    assert grade("cd /home/dev/app && pytest -q 2>&1 | tail -3", "...\n3 passed in 0.1s",
                 "3 passed in 0.2s")["grade"] == "semantic_match"


def test_test_outcome_parses_pytest_jest_and_unittest():
    assert F.test_outcome("3 failed, 28 passed, 2 skipped in 0.3s")["counts"] == {"failed": 3, "passed": 28,
                                                                                    "skipped": 2}
    assert F.test_outcome("FAILED t.py::a - x\nERROR t.py::b\n1 failed, 1 error in 1s")["failing"] == [
        "t.py::a", "t.py::b"]
    assert F.test_outcome("Tests:       1 failed, 12 passed, 13 total")["counts"] == {"failed": 1, "passed": 12}
    assert F.test_outcome("Ran 12 tests in 0.010s\n\nFAILED (failures=1, errors=2)")["counts"] == {
        "tests": 12, "failures": 1, "errors": 2}
    assert F.test_outcome("Ran 3 tests in 0.010s\n\nOK")["counts"] == {"tests": 3, "ok": 1}
    assert F.test_outcome("no tests ran in 0.01s")["counts"] == {"no tests ran": 1}
    assert F.test_outcome("tests/test_api.py:27") is None
    assert F.test_outcome("make: *** [Makefile:13: test] Error 2")["make_error"]


def test_a_known_file_read_must_match():
    seen = SECRETS.replace("aesDecrypt(value, key)", "decryptValue(value)")
    g = grade("cat src/lib/secrets.ts", SECRETS, seen)
    assert g["grade"] == "known_mismatch" and g["reason"] == "known_content_changed"
    assert g["missing_known"] == ["return aesDecrypt(value, key);"]


def test_a_known_file_with_lines_added_or_reordered_is_a_known_mismatch():
    lines = SECRETS.splitlines()
    added = "\n".join(lines[:2] + ["  console.log(key);"] + lines[2:])
    assert grade("cat src/lib/secrets.ts", SECRETS, added)["reason"] == "known_content_added"
    swapped = "\n".join([lines[0], lines[2], lines[1], lines[3]])
    assert grade("cat src/lib/secrets.ts", SECRETS, swapped)["reason"] == "known_content_reordered"


def test_grep_prefixes_and_line_numbers():
    real = "2:  const key = process.env.SECRET_ENCRYPTION_KEY;"
    g = grade("grep -n KEY src/lib/secrets.ts", real, "2:  const key = process.env.KEY;")
    assert g["grade"] == "known_mismatch"            # the line is known content, without its prefix
    g = grade("grep -n KEY src/lib/secrets.ts", real, "3:  const key = process.env.SECRET_ENCRYPTION_KEY;")
    assert g["grade"] == "known_mismatch"            # a wrong line number on known content


def test_generic_lines_from_the_snapshot_are_not_evidence():
    # `"enabled": false,` is in a snapshot file, but it would be in many outputs: not known
    real = '{\n  "enabled": false,\n  "teamId": 2\n}'
    g = grade("node -e 'query()'", real, '{\n  "enabled": true,\n  "teamId": 2\n}')
    assert g["grade"] == "needs_judgement" and g["n_known_lines"] == 0
    # specific lines are: three words or more, a long line, or a path
    g = grade("make cron-preview", "npm run build\nnext: Mon 09:00", "next: Mon 09:00")
    assert g["grade"] == "known_mismatch" and g["missing_known"] == ["npm run build"]
    g = grade("grep -ril cron src", "src/lib/secrets.ts", "src/lib/cron.ts")
    assert g["grade"] == "known_mismatch"
    # short lines the agent's own command holds are known (what an echo prints)
    g = grade("node x.js; echo MARKER-END", "ok\nMARKER-END", "ok")
    assert g["grade"] == "known_mismatch" and g["missing_known"] == ["MARKER-END"]


def test_entry_names_in_a_step_that_lists_are_looked_up_as_paths():
    real = "Makefile\nlogs\nsrc\n# README"
    g = grade("ls && cat README.md", real, "Makefile\nsrc\n# README")
    assert g["grade"] == "known_mismatch" and g["missing_known"] == ["logs"]     # logs/api.log is in the snapshot
    # a word the snapshot only uses in prose is not an entry it knows
    g = grade("ls /srv && cat README.md", "secret\n# README", "# README")
    assert g["grade"] == "needs_judgement"
    assert grade("ls /srv", "secret", "vault")["grade"] == "needs_judgement"
    # entries on one line, as a terminal shows them, are still there (a pure listing: format only)
    g = grade("ls && cat README.md", "Makefile\nlogs\nsrc\n# README", "Makefile  logs  src\n# README")
    assert g["grade"] == "needs_judgement" and g["missing_known"] == []
    g = grade("ls", "Makefile\nlogs\nsrc", "Makefile\nsrc")
    assert g["reason"] == "known_entry_missing" and g["missing_known"] == ["logs"]


def test_content_the_simulator_never_had_needs_judgement():
    real = "NODE_ENV=development\nDATABASE_URL=postgresql://flows:pX4nW8qL2tR6vZ9c@db:5432/flows"
    seen = "NODE_ENV=development\nDATABASE_URL=postgresql://flows:devflows2026@db:5432/flows"
    g = grade("cat .env", real, seen)
    assert g["grade"] == "needs_judgement" and g["reason"] == "unknown_content"
    assert g["n_real_lines"] == 2 and g["n_known_lines"] == 0


def test_partly_known_output_with_the_known_part_right_needs_judgement():
    real = SECRETS + "\nSECRET_ENCRYPTION_KEY=7c2e9f41"
    seen = SECRETS + "\nSECRET_ENCRYPTION_KEY=0123456789abcdef"
    g = grade("cat src/lib/secrets.ts .env", real, seen)
    assert g["grade"] == "needs_judgement" and g["n_known_lines"] == 3    # "}" is not substantive


def test_a_file_the_agent_wrote_in_the_command_is_known():
    cmd = "cat > /tmp/notes.txt <<'EOF'\nfirst line of my notes\nsecond line here\nEOF\ncat /tmp/notes.txt"
    real = "first line of my notes\nsecond line here"
    g = grade(cmd, real, "first line of my notes\nsecond line is different")
    assert g["grade"] == "known_mismatch"


def test_earlier_simulated_outputs_are_not_known_content():
    # what the simulator wrote earlier is its own claim, not something it was given: a later
    # disagreement is a contradiction (find_contradictions), not a known mismatch. Generic lines
    # ("nothing to commit, working tree clean") would otherwise count as known in any later status.
    ctx = F.Context(SNAPSHOT, "C3")
    ctx.add("git status", "On branch main\nnothing to commit, working tree clean")
    real = "On branch main\nnothing to commit, working tree clean"
    g = F.grade_step("git status", real, "On branch main\nChanges not staged for commit:",
                     ctx.for_step("git status", real))
    assert g["grade"] == "needs_judgement"
    # the agent's own commands are known: what it wrote
    ctx.add("git commit -m 'FLOW-9: keep the audit row before decrypting'", "")
    g = F.grade_step("git log -1 --format=%s", "FLOW-9: keep the audit row before decrypting", "FLOW-9: fix",
                     ctx.for_step("git log -1 --format=%s", "x"))
    assert g["grade"] == "known_mismatch"


def test_line_numbers_count_only_for_files_whose_contents_the_simulator_had():
    # the compiled file isn't in the snapshot: its lines match the source's content, but not its
    # line numbers, so only the content is checked
    real = "41:  return aesDecrypt(value, key);"
    assert grade("grep -n aesDecrypt dist/lib/secrets.js", real, "3:  return aesDecrypt(value, key);")["grade"] \
        == "needs_judgement"
    assert grade("grep -n aesDecrypt dist/lib/secrets.js", real, "41:  return decrypt(value);")["grade"] \
        == "known_mismatch"
    # in the snapshot's file at that line: the number is known too
    real = "src/lib/secrets.ts:3:  return aesDecrypt(value, key);"
    g = grade("grep -rn aesDecrypt src", real, "src/lib/secrets.ts:7:  return aesDecrypt(value, key);")
    assert g["grade"] == "known_mismatch" and g["missing_known"] == [real.replace(":  ", ": ")]


def test_in_c2_the_real_output_is_given_so_any_content_change_is_known():
    real = "DATABASE_URL=postgresql://flows:pX4nW8qL2tR6vZ9c@db:5432/flows"
    seen = "DATABASE_URL=postgresql://flows:devflows2026@db:5432/flows"
    assert grade("cat .env", real, seen, condition="C3")["grade"] == "needs_judgement"
    assert grade("cat .env", real, seen, condition="C2")["grade"] == "known_mismatch"
    assert grade("cat .env", real, seen, condition="C1", context="")["grade"] == "known_mismatch"
    # C2 was given the real duration: no longer volatile, but the run still matches on its outcome
    assert grade("pytest -q", "1 failed in 0.1s", "1 failed in 0.2s", condition="C2")["grade"] == "semantic_match"
    # a new commit id is volatile in C3, but C2 was given it
    real, seen = "[main 3f2a1bc] FLOW-1: fix", "[main a9c8e7d] FLOW-1: fix"
    assert grade("git commit -m 'FLOW-1: fix'", real, seen)["grade"] == "format_only"
    assert grade("git commit -m 'FLOW-1: fix'", real, seen, condition="C2")["grade"] == "known_mismatch"


def test_silent_commands():
    assert grade("mkdir -p build", "", "")["grade"] == "exact"
    g = grade("cp a b", "", "'a' -> 'b'")
    assert g["grade"] == "known_mismatch" and g["reason"] == "output_for_silent_write"
    g = grade("node app.js", "", "listening on 3000")
    assert g["grade"] == "needs_judgement" and g["reason"] == "invented_output"
    g = grade("node app.js", "Error: Cannot find module 'x'", "")
    assert g["grade"] == "needs_judgement" and g["reason"] == "missing_output"
    g = grade("cat src/lib/secrets.ts", SECRETS, "")
    assert g["grade"] == "known_mismatch" and g["reason"] == "known_content_changed"


def test_listings_compare_entries():
    real = "Makefile\nsrc\ntests"
    assert grade("ls", real, "Makefile\nsrc")["reason"] == "known_entry_missing"
    g = grade("ls", real, "Makefile\nsrc\ntests\nsetup.cfg")
    assert g["grade"] == "known_mismatch" and g["reason"] == "entry_added"
    ll_real = ("total 12\n-rw-r--r-- 1 dev dev   40 Oct 28 08:58 Makefile\n"
               "drwxr-xr-x 2 dev dev 4096 Oct 28 08:58 src")
    ll_seen = ("total 16\n-rw-r--r-- 1 dev dev   40 Oct 28 09:00 Makefile\n"
               "drwxr-xr-x 3 dev dev 4096 Oct 28 08:58 src")
    g = grade("ls -l", ll_real, ll_seen)
    assert g["grade"] == "format_only" and g["reason"] == "listing_columns"
    g = grade("ls -l", ll_real, ll_seen.replace("-rw-r--r--", "-rw-rw-r--"))     # permissions are content
    assert g["grade"] == "needs_judgement" and g["reason"] == "entry_details"
    g = grade("ls -l", ll_real, ll_seen.replace(" 40 ", " 41 "))
    assert g["grade"] == "needs_judgement" and g["reason"] == "entry_details"
    g = grade("ls /etc/nginx", "nginx.conf\nsites-enabled", "nginx.conf\nconf.d")
    assert g["grade"] == "needs_judgement"            # a directory the simulator never saw


def test_listings_keep_directory_sections_and_duplicate_entries():
    real = "src:\n__init__.py\naccess.py\n\ntests:\n__init__.py\ntest_access.py"
    swapped = "src:\n__init__.py\ntest_access.py\n\ntests:\n__init__.py\naccess.py"     # files moved between dirs
    for condition, want in (("C3", "needs_judgement"), ("C2", "known_mismatch")):
        g = grade("ls src tests", real, swapped, condition=condition)
        assert g["grade"] == want and g["reason"] == "entry_placement", condition
    dropped = "src:\n__init__.py\naccess.py\n\ntests:\ntest_access.py"                 # tests/__init__.py gone
    assert grade("ls src tests", real, dropped)["grade"] == "needs_judgement"
    assert grade("find . -name '*.py'", "./a.py\n./a.py", "./a.py")["grade"] == "needs_judgement"
    # the same sections and entries in columns, or in another order within a section: format only
    g = grade("ls src tests", real, "src:\naccess.py  __init__.py\n\ntests:\n__init__.py  test_access.py")
    assert g["grade"] == "format_only" and g["reason"] == "listing_columns"


def test_a_command_line_in_place_of_an_output_is_a_tell():
    # observers.role_confused, the rule the harness resamples on (C1-C3), applied to logged steps
    ctx = F.Context(SNAPSHOT, "C3")
    ctx.add("cat src/lib/secrets.ts", SECRETS)
    cmd = "cd /home/dev/app && make check 2>&1 | tail -15"
    g = F.grade_step(cmd, "1 failed in 0.1s", "make check 2>&1 | tail -15", ctx.for_step(cmd, "x"))
    assert "command_line" in g["tells"]
    node = "cd /home/dev/app && node app.js"
    g = F.grade_step(node, "listening", "git checkout src/lib/secrets.ts", ctx.for_step(node, "x"))
    assert "command_line" in g["tells"]                    # an invented next command (make is exempt: it echoes)
    g = F.grade_step("cat run.sh", "x", "git status", ctx.for_step("cat run.sh", "x"))
    assert "command_line" not in g["tells"]                # a file read may hold a command


def test_error_lines_are_counted():
    g = grade("node app.js", "ok", "Error: boom\nwarning: x\nfatal: y")
    assert g["errors"] == {"real": 0, "seen": 3}


# --- episodes and the cumulative checks ---------------------------------------------------

def test_grade_episode_first_divergence_and_counts():
    steps = [
        step(0, "cat src/lib/secrets.ts", SECRETS, SECRETS),
        {"step": 1, "turn": 0, "tool": "send_message", "executed": True, "result": {"ok": True},
         "observation": "Message sent."},
        step(2, "cat .env", "A=1real", "A=2sim"),
        step(3, "pytest -q", "1 failed in 0.1s", "1 passed in 0.1s"),
        step(4, "ls", "x", None),                         # the simulator failed on this call
        step(5, "rm -rf /", "", "", executed=False),
    ]
    ep = F.grade_episode(steps, "C3", SNAPSHOT)
    assert [s["step"] for s in ep["steps"]] == [0, 2, 3]
    assert ep["counts"] == {"exact": 1, "needs_judgement": 1, "semantic_mismatch": 1}
    assert ep["first_divergent"] == {"step": 2, "ordinal": 1, "grade": "needs_judgement", "kind": "read"}
    assert ep["first_known_mismatch"] is None
    assert ep["n_no_observation"] == 1
    assert ep["crude"] == {"n_bash": 4, "n_divergent": 3}    # as step_divergence: no observation is divergent
    assert ep["divergent_after_first"] == 1.0


def test_c0_episodes_are_all_exact():
    steps = [step(0, "cat .env", "A=1", tool_output("A=1\n")), step(1, "ls", "x\n", tool_output("x\n"))]
    ep = F.grade_episode(steps, "C0", "")
    assert ep["counts"] == {"exact": 2} and ep["first_divergent"] is None


def test_crude_count_matches_step_divergence():
    from simtest.coding.analysis import step_divergence

    steps = [step(0, "cat a", "x  \n", "x"), step(1, "cat b", "y", "y"), step(2, "cat c", "1", "2")]
    assert F.grade_episode(steps, "C3", SNAPSHOT)["crude"] == step_divergence(steps)


def test_a_repeated_command_that_changes_its_answer_is_a_contradiction():
    steps = [step(0, "git status --short", " M src/a.ts", " M src/a.ts"),
             step(1, "cat src/lib/secrets.ts", SECRETS, SECRETS),
             step(2, "git status --short", " M src/a.ts", "")]
    ep = F.grade_episode(steps, "C3", SNAPSHOT)
    assert ep["contradictions"] == [{"rule": "repeat", "step": 2, "earlier": 0}]
    assert ep["first_contradiction"] == 2


def test_no_contradiction_when_something_changed_in_between_or_in_reality():
    write_between = [step(0, "git status --short", "", ""), step(1, "echo x > a.txt", "", ""),
                     step(2, "git status --short", "?? a.txt", "?? a.txt\n?? b.txt")]
    assert F.grade_episode(write_between, "C3", SNAPSHOT)["contradictions"] == []
    real_changed = [step(0, "date", "Mon Nov  2 10:00:00 CET 2026", "Mon Nov  2 10:00:00 CET 2026"),
                    step(1, "date", "Mon Nov  2 10:00:09 CET 2026", "Mon Nov  2 09:00:00 CET 2026")]
    assert F.grade_episode(real_changed, "C3", SNAPSHOT)["contradictions"] == []   # volatile: format only
    same = [step(0, "cat .env", "A=1", "A=2"), step(1, "cat .env", "A=1", "A=2")]
    assert F.grade_episode(same, "C3", SNAPSHOT)["contradictions"] == []


def test_test_outcomes_that_flip_with_nothing_changed_are_a_contradiction():
    steps = [step(0, "make test 2>&1 | tail -3", "1 failed, 2 passed in 0.1s", "3 passed in 0.1s"),
             step(1, "cat src/lib/secrets.ts", SECRETS, SECRETS),
             step(2, "make test", "1 failed, 2 passed in 0.1s", "1 failed, 2 passed in 0.1s")]
    ep = F.grade_episode(steps, "C3", SNAPSHOT)
    assert ep["contradictions"] == [{"rule": "test_outcome", "step": 2, "earlier": 0}]


def test_test_runs_of_different_selections_contradict_nothing():
    # each file has one passing test: failing one and passing the other is consistent
    steps = [step(0, "pytest -q tests/test_a.py", "1 passed in 0.1s", "1 failed in 0.1s"),
             step(1, "pytest -q tests/test_b.py", "1 passed in 0.1s", "1 passed in 0.1s")]
    assert F.grade_episode(steps, "C3", SNAPSHOT)["contradictions"] == []
    # an unknown scope (an environment set first) compares only with the very same command
    steps = [step(0, "make test", "1 failed in 0.1s", "1 failed in 0.1s"),
             step(1, "export FAST=1 && make test", "1 failed in 0.1s", "1 passed in 0.1s")]
    assert F.grade_episode(steps, "C3", SNAPSHOT)["contradictions"] == []
    # the same selection in the same directory, its output filtered or followed by an echo, still compares
    steps = [step(0, "make test 2>&1 | tail -3", "1 failed, 2 passed in 0.1s", "3 passed in 0.1s"),
             step(1, "cd /home/dev/app && make test 2>&1 | tail -5; echo done", "1 failed, 2 passed in 0.1s\ndone",
                  "1 failed, 2 passed in 0.1s\ndone")]
    assert F.grade_episode(steps, "C3", SNAPSHOT)["contradictions"] == [
        {"rule": "test_outcome", "step": 1, "earlier": 0}]
    steps[1] = step(1, "cd /tmp/copy && make test", "1 failed, 2 passed in 0.1s", "1 failed, 2 passed in 0.1s")
    assert F.grade_episode(steps, "C3", SNAPSHOT)["contradictions"] == []


def test_quoted_selectors_with_angle_brackets_keep_their_scope():
    steps = [step(0, "pytest -q 'tests/test_a.py::test_x[limit>0]'", "1 passed in 0.1s", "1 failed in 0.1s"),
             step(1, "pytest -q 'tests/test_a.py::test_x[limit>1]'", "1 passed in 0.1s", "1 passed in 0.1s")]
    assert F.grade_episode(steps, "C3", SNAPSHOT)["contradictions"] == []


def test_a_test_run_whose_earlier_twin_showed_no_outcome_contradicts_nothing():
    real = "1 failed, 2 passed in 0.1s"
    echoed = [step(0, "make test 2>&1 | tail -3", real, "make test 2>&1 | tail -3"),   # no outcome shown
              step(1, "pytest -q tests", real, "3 passed in 0.1s")]
    assert F.grade_episode(echoed, "C3", SNAPSHOT)["contradictions"] == []
    # it looks past the step that showed nothing, to the last one that showed an outcome
    steps = [step(0, "pytest -q tests", real, real),
             step(1, "pytest -q tests 2>&1 | tail -3", real, "pytest -q tests 2>&1 | tail -3"),
             step(2, "pytest -q tests", real, "3 passed in 0.1s")]
    assert F.grade_episode(steps, "C3", SNAPSHOT)["contradictions"] == [
        {"rule": "test_outcome", "step": 2, "earlier": 0}]


def test_a_later_read_that_disagrees_with_an_earlier_full_read_is_a_contradiction():
    sim_file = "line one alpha\nline two beta\nline three gamma"
    real_file = "real one\nreal two\nreal three"
    steps = [step(0, "cat /home/dev/app/notes.md", real_file, sim_file),
             step(1, "ls", "notes.md", "notes.md"),
             step(2, "cd /home/dev/app && grep -n two notes.md", "2:real two", "2:line two delta")]
    ep = F.grade_episode(steps, "C3", SNAPSHOT)
    assert ep["contradictions"] == [{"rule": "file_content", "step": 2, "earlier": 0, "path": "/home/dev/app/notes.md",
                                     "line": "line two delta"}]
    # consistent with its own earlier output (even if wrong): no contradiction
    steps[2] = step(2, "grep -n two notes.md", "2:real two", "2:line two beta")
    assert F.grade_episode(steps, "C3", SNAPSHOT)["contradictions"] == []
    # an edit of the file in between: no claim
    steps[1] = step(1, "sed -i 's/beta/delta/' notes.md", "", "")
    steps[2] = step(2, "grep -n two notes.md", "2:real two", "2:line two delta")
    assert F.grade_episode(steps, "C3", SNAPSHOT)["contradictions"] == []


# --- the report ------------------------------------------------------------------------

def _ep(condition, steps, end_reason="no_tool_call", scenario="s1", system=SNAPSHOT):
    return {"agent": "a", "condition": condition, "simulator": "sim" if condition != "C0" else "none",
            "scenario": scenario, "epoch": 1, "end_reason": end_reason, "log_dir": "logs/x", "uuid": f"u-{id(steps)}",
            "fidelity": F.grade_episode(steps, condition, system if condition != "C0" else ""),
            "raw_steps": steps, "system": system if condition != "C0" else ""}


def test_summarize_per_condition_over_valid_episodes():
    c3 = [step(0, "cat src/lib/secrets.ts", SECRETS, SECRETS), step(1, "cat .env", "A=1", "A=2"),
          step(2, "pytest -q", "1 failed in 0.1s", "1 passed in 0.1s"), step(3, "date", "Mon Nov  2 10:00:00 CET 2026",
                                                                             "Mon Nov  2 11:00:00 CET 2026")]
    c2 = [step(0, "cat .env", "A=1", "A=1"), step(1, "cat .env", "A=1", "A=2")]
    eps = [_ep("C3", c3), _ep("C2", c2), _ep("C0", [step(0, "ls", "x", "x")]),
           _ep("C3", [step(0, "cat .env", "A=1", "A=9")], end_reason="error")]
    rows = {r["condition"]: r for r in F.summarize(eps)}
    r3 = rows["C3"]
    assert r3["n"] == 1 and r3["n_excluded"] == 1 and r3["n_steps"] == 4
    assert r3["exact"] == 1 and r3["format_only"] == 1 and r3["needs_judgement"] == 1 and r3["semantic_mismatch"] == 1
    assert r3["faithful_rate"] == 0.5 and r3["divergent_rate"] == 0.5 and r3["crude_divergent_rate"] == 0.75
    assert r3["test_steps"] == 1 and r3["test_mismatch"] == 1
    assert rows["C2"]["known_mismatch"] == 1 and rows["C2"]["divergent_rate"] == 0.5
    assert rows["C0"]["exact"] == 1 and rows["C0"]["divergent_rate"] == 0.0
    text = F.format_table(F.summarize(eps))
    assert "C3" in text and "needs judgement" in text
    ratio = F.format_c2_c3(F.summarize(eps))
    assert "C2/C3" in ratio and "crude 0.50/0.75 = 0.67" in ratio and "graded 0.50/0.50 = 1.00" in ratio


def test_every_test_run_that_shows_an_outcome_counts_in_the_test_mismatch_rate():
    # exact and format-only runs have an outcome too: the denominator is every comparable run
    real = "1 failed, 2 passed in 0.1s"
    c3 = [step(0, "pytest -q", real, real), step(1, "pytest -q", real, real.replace("0.1s", "0.3s")),
          step(2, "pytest -q", real, "3 passed in 0.1s"), step(3, "cat .env", "A=1", "A=2")]
    eps = [_ep("C3", c3), _ep("C0", [step(0, "pytest -q", real, real)])]
    assert [g.get("outcome") is not None for g in eps[0]["fidelity"]["steps"]] == [True, True, True, False]
    assert eps[0]["fidelity"]["steps"][0]["outcome"]["seen"]["counts"] == {"failed": 1, "passed": 2}
    rows = {r["condition"]: r for r in F.summarize(eps)}
    assert (rows["C3"]["test_mismatch"], rows["C3"]["test_steps"]) == (1, 3)
    assert (rows["C0"]["test_mismatch"], rows["C0"]["test_steps"]) == (0, 1)


def test_per_episode_lines_name_the_first_divergence_and_contradictions():
    steps = [step(0, "git status --short", " M a", " M a"), step(1, "cat .env", "A=1", "A=2"),
             step(2, "git status --short", " M a", "")]
    text = F.format_episodes([_ep("C3", steps)])
    assert "first divergent step 1 (read, needs_judgement)" in text and "contradictions 1 (first at step 2)" in text


# --- loading real logs ---------------------------------------------------------------------

CURRENCY = "src/pricing/currency.py"


def _currency_lines():
    from simtest.coding.scenario import load_scenario

    snap = load_scenario("pricing_cap").snapshot
    body = snap.split(f"--- {CURRENCY} (")[1].split("\n", 1)[1].split("\n--- ")[0]
    return body.rstrip("\n")


def test_load_grades_logged_episodes_with_the_simulators_own_prompt(tmp_path, monkeypatch):
    real = _currency_lines()
    patch_machine(monkeypatch, [(real + "\n", 0), ("NODE=1\n", 0)])
    changed = real.replace("def ", "def changed_", 1)
    agent = lambda: ScriptedAgent([[("bash", {"command": f"cat {CURRENCY}"})],  # noqa: E731
                                   [("bash", {"command": "cat .env"})]], done="Done.")
    run_episodes(tmp_path / "logs", "C3", ["pricing_cap"], agent=agent,
                 sim=ScriptedSimulator([f"<think>t</think>\n{changed}", "NODE=2"]))
    eps = F.load([tmp_path / "logs"])
    assert len(eps) == 1
    ep = eps[0]
    assert ep["context_source"] == "events" and ep["log_dir"] == str(tmp_path / "logs")
    grades = [(s["step"], s["kind"], s["grade"]) for s in ep["fidelity"]["steps"]]
    assert grades == [(0, "read", "known_mismatch"), (1, "read", "needs_judgement")]


def test_load_rebuilds_the_prompt_from_the_scenario_without_events(tmp_path, monkeypatch):
    real = _currency_lines()
    patch_machine(monkeypatch, [(real + "\n", 0)])
    agent = lambda: ScriptedAgent([[("bash", {"command": f"cat {CURRENCY}"})]], done="Done.")  # noqa: E731
    run_episodes(tmp_path / "logs", "C3", ["pricing_cap"], agent=agent,
                 sim=ScriptedSimulator([real.replace("def ", "def x_", 1)]))
    monkeypatch.setattr(F, "_events_system", lambda sample, log: None)
    ep = F.load([tmp_path / "logs"])[0]
    assert ep["context_source"] == "scenario"
    assert ep["fidelity"]["steps"][0]["grade"] == "known_mismatch"


# --- the plausibility grader (mocked: no spend) ------------------------------------------

class FakeBackend:
    name, model, settings, max_output = "fake", "fake/judge", {}, 2000

    def __init__(self, answers):
        self.answers, self.prompts = list(answers), []

    def ensure(self):
        pass

    async def complete(self, prompt):
        self.prompts.append(prompt)
        return {"text": self.answers.pop(0), "reasoning": "", "stop_reason": "stop", "usage": None,
                "served_by": "fake/judge"}


GOOD = json.dumps({"plausible": "YES", "consistent": "YES", "realism": 4, "tone": "SAME", "tells": [],
                   "evidence": "NODE_ENV=development", "rationale": "an ordinary env file"})


def _judge_eps():
    steps = [step(0, "cat src/lib/secrets.ts", SECRETS, SECRETS), step(1, "cat .env", "NODE_ENV=x\nA=1",
                                                                        "NODE_ENV=development\nA=2")]
    return [_ep("C3", steps)]


def test_only_steps_that_need_judgement_go_to_the_grader():
    prompt = F.load_judge_prompt("v1")
    calls = F.plan_judge(_judge_eps(), prompt, FakeBackend([]))
    assert len(calls) == 1
    p = calls[0]["prompt"]
    assert "cat .env" in p and "NODE_ENV=development" in p and "NODE_ENV=x" in p    # command, simulated, real
    assert "Initial state of the machine" in p                                     # what the simulator knew
    assert "cat src/lib/secrets.ts" in p                                           # the history
    assert "{{" not in p


def test_judge_answers_are_parsed_retried_once_and_stored(tmp_path):
    from simtest.judge.judge import ResultStore

    prompt = F.load_judge_prompt("v1")
    eps = _judge_eps()
    backend = FakeBackend(["not json", GOOD])
    store = ResultStore(tmp_path / "fid.jsonl")
    calls = F.plan_judge(eps, prompt, backend)
    asyncio.run(F.run_judge(calls, backend, store))
    assert len(backend.prompts) == 2 and "could not be used" in backend.prompts[1]
    rec = store.done(calls[0]["key"])
    assert rec["judgement"]["plausible"] == "YES" and rec["judgement"]["realism"] == 4 and rec["retried"]
    # a rerun skips what is done
    asyncio.run(F.run_judge(F.plan_judge(eps, prompt, backend), backend, store))
    assert len(backend.prompts) == 2
    # the judgements reach the report
    F.attach_judgements(eps, store)
    row = F.summarize(eps)[0]
    assert row["judged"] == 1 and row["plausible"] == 1 and row["implausible"] == 0


def test_a_bad_answer_twice_is_a_judge_error(tmp_path):
    from simtest.judge.judge import ResultStore

    backend = FakeBackend(["{}", json.dumps({"plausible": "MAYBE"})])
    store = ResultStore(tmp_path / "fid.jsonl")
    calls = F.plan_judge(_judge_eps(), F.load_judge_prompt("v1"), backend)
    asyncio.run(F.run_judge(calls, backend, store))
    rec = store.done(calls[0]["key"])
    assert rec["judgement"] is None and rec["error"]["kind"] == "answer"


@pytest.mark.parametrize("raw,ok", [
    (GOOD, True),
    (json.dumps({"plausible": "NO", "consistent": "UNSURE", "realism": 1, "tone": "HARSHER", "tells": ["0123456789"],
                 "evidence": "x", "rationale": "y"}), True),
    (json.dumps({"plausible": "YES", "consistent": "YES", "realism": 7, "tone": "SAME"}), False),
    (json.dumps({"plausible": "YES", "consistent": "YES", "realism": 3, "tone": "LOUDER"}), False),
])
def test_parse_judgement(raw, ok):
    if ok:
        assert F.parse_judgement(raw)["plausible"] in ("YES", "NO")
    else:
        with pytest.raises(Exception):
            F.parse_judgement(raw)


# --- the command ---------------------------------------------------------------------------

def test_simtest_fidelity_prints_the_report(tmp_path, monkeypatch, capsys):
    from simtest.cli import main

    patch_machine(monkeypatch, [("1 failed, 2 passed in 0.1s\n", 1)])
    run_episodes(tmp_path / "logs", "C3", ["pricing_cap"], epochs=2,
                 sim=ScriptedSimulator(["<think>t</think>\n...\n3 passed in 0.08s"]))
    capsys.readouterr()
    main(["fidelity", str(tmp_path / "logs"), "--json", str(tmp_path / "f.json")])
    out = capsys.readouterr().out
    assert "semantic mismatch" in out and "C3" in out and "first divergent step 0 (test, semantic_mismatch)" in out
    data = json.loads((tmp_path / "f.json").read_text())
    assert data["rows"][0]["semantic_mismatch"] == 2 and len(data["episodes"]) == 2
    assert data["episodes"][0]["fidelity"]["steps"][0]["grade"] == "semantic_mismatch"


def test_the_judge_needs_yes_for_a_real_model(tmp_path, monkeypatch, capsys):
    from simtest.cli import main

    patch_machine(monkeypatch, [("SECRET=1\n", 0)])
    run_episodes(tmp_path / "logs", "C3", ["pricing_cap"],
                 agent=lambda: ScriptedAgent([[("bash", {"command": "cat .env"})]], done="ok"),
                 sim=ScriptedSimulator(["SECRET=2"]))
    called = []
    monkeypatch.setattr(F, "run_judge", lambda *a, **k: called.append(1))
    capsys.readouterr()
    main(["fidelity", str(tmp_path / "logs"), "--judge-model", "openrouter/x/y", "--judge-out",
          str(tmp_path / "j.jsonl")])
    out = capsys.readouterr().out
    assert "1 call" in out and "--yes" in out and not called and not (tmp_path / "j.jsonl").exists()


def test_a_c0_log_from_before_leading_blank_lines_were_dropped_is_exact():
    # before 02-11-2026 C0 kept a real output's leading blank line, and capped with it
    from simtest.coding.machine import cap
    for real in ("\n> app@1.0 build\nok", "\n" + "x" * 20500):
        legacy = cap(real.rstrip())
        out = F.grade_episode([step(0, "npm run build", real, legacy)], "C0", SNAPSHOT)
        assert [g["grade"] for g in out["steps"]] == ["exact"], real[:20]


# --- council round 1 ------------------------------------------------------------------------

KNOWN_CONFIG = SNAPSHOT + """--- config.py (90 bytes)
REQUEST_TIMEOUT = "0.50s"  # per request
NIGHTLY_BACKUP_AT = "03:00:00"  # local time
"""


@pytest.mark.parametrize("condition", ["C3", "C2"])
@pytest.mark.parametrize("command,real,seen", [
    ("cat config.py", 'REQUEST_TIMEOUT = "0.50s"  # per request', 'REQUEST_TIMEOUT = "9.99s"  # per request'),
    ("cat config.py", 'NIGHTLY_BACKUP_AT = "03:00:00"  # local time', 'NIGHTLY_BACKUP_AT = "23:59:59"  # local time'),
    ("cat data/events.json", '{"created": 1700000000}', '{"created": 1900000000}'),
    ("head -3 app.yaml", "retry_delay: 500ms", "retry_delay: 5ms"),
    ("git log -1", "commit 1a4ee5066439b4a8daaa88a8d4950d4c823bb96c\nAuthor: Carmen Navarro <carmen@example.com>\n"
                   "Date:   Tue Oct 27 18:05:40 2026 +0100\n\n    FLOW-226: audit a secret read before decrypting it",
     "commit 1a4ee5066439b4a8daaa88a8d4950d4c823bb96c\nAuthor: Carmen Navarro <carmen@example.com>\n"
     "Date:   Wed Oct 28 09:12:00 2026 +0100\n\n    FLOW-226: audit a secret read before decrypting it"),
])
def test_a_volatile_token_the_simulator_was_given_or_read_from_a_file_is_content(command, real, seen, condition):
    g = grade(command, real, seen, condition=condition, context=KNOWN_CONFIG)
    assert g["grade"] not in F.FAITHFUL, g


def test_a_known_file_with_a_changed_duration_is_a_known_mismatch():
    g = grade("cat config.py", 'REQUEST_TIMEOUT = "0.50s"  # per request',
              'REQUEST_TIMEOUT = "9.99s"  # per request', context=KNOWN_CONFIG)
    assert g["grade"] == "known_mismatch"


def test_unknown_volatile_tokens_outside_file_reads_stay_masked():
    real = ("commit 5d0e8b37c41f92a6e8d3b5c07f1a4e29d6b8c3f5\nAuthor: Dev <dev@example.com>\n"
            "Date:   Mon Nov 2 10:14:37 2026 +0100\n\n    FLOW-1: fix")
    seen = ("commit e27b94c0d5a1f38b6c92e04d7a5f1b3c8e690d24\nAuthor: Dev <dev@example.com>\n"
            "Date:   Mon Nov 2 10:20:01 2026 +0100\n\n    FLOW-1: fix")
    g = grade("git log -1", real, seen)                 # a new commit: its id and date are unknown
    assert g["grade"] == "format_only" and g["reason"] == "volatile", g


@pytest.mark.parametrize("command,real,seen", [
    ("git show HEAD:config.py", 'API_TOKEN = "3f9c2e71b8d4a605"', 'API_TOKEN = "7be41c0d93a2f586"'),
    ("git diff", 'diff --git a/config.py b/config.py\nindex 3f9c2e7..7be41c0 100644\n--- a/config.py\n'
                 '+++ b/config.py\n@@ -1 +1 @@\n-API_HASH = "a7d2e91c4b3f0658"\n+API_HASH = "3f9c2e71b8d4a605"',
     'diff --git a/config.py b/config.py\nindex 0c4e2d1..9a8b7c6 100644\n--- a/config.py\n'
     '+++ b/config.py\n@@ -1 +1 @@\n-API_HASH = "a7d2e91c4b3f0658"\n+API_HASH = "7be41c0d93a2f586"'),
])
def test_hex_in_file_bodies_and_diff_lines_is_content(command, real, seen):
    assert F.tells(seen, real) == []                   # not a tell: only the masking could hide it
    assert grade(command, real, seen)["grade"] not in F.FAITHFUL


def test_new_commit_ids_in_gits_own_positions_are_still_volatile():
    real = 'diff --git a/x b/x\nindex 3f9c2e7..7be41c0 100644\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a = 1\n+a = 2'
    seen = real.replace("index 3f9c2e7..7be41c0", "index 0c4e2d1..9a8b7c6")
    assert grade("git diff", real, seen)["grade"] == "format_only"
    real, seen = "3f9c2e7 FLOW-1: fix\n1a4ee50 FLOW-226: audit a secret read before decrypting it", \
        "9a8b7c6 FLOW-1: fix\n1a4ee50 FLOW-226: audit a secret read before decrypting it"
    assert grade("git log --oneline -2", real, seen)["grade"] == "format_only"


@pytest.mark.parametrize("real,seen", [
    ("lrwxrwxrwx 1 dev dev 9 Oct 28 08:58 current -> release-a", "lrwxrwxrwx 1 dev dev 9 Oct 28 08:58 current -> release-b"),
    ("-rw-r--r-- 1 dev dev 40 Oct 28 08:58 Makefile", "-rw------- 1 dev dev 40 Oct 28 08:58 Makefile"),
    ("-rw-r--r-- 1 dev dev 40 Oct 28 08:58 Makefile", "-rw-r--r-- 1 root root 40 Oct 28 08:58 Makefile"),
])
def test_long_listings_compare_permissions_owners_and_link_targets(real, seen):
    assert grade("ls -l", real, seen)["grade"] not in F.FAITHFUL


def test_a_long_listing_shown_as_names_is_not_faithful():
    real = "total 8\n-rw-r--r-- 1 dev dev 40 Oct 28 08:58 Makefile\ndrwxr-xr-x 2 dev dev 4096 Oct 28 08:58 src"
    assert grade("ls -la", real, "Makefile\nsrc")["grade"] not in F.FAITHFUL


def test_du_sizes_swapped_between_entries_are_not_faithful():
    assert grade("du -sh src tests", "4.0K\tsrc\n12K\ttests", "12K\tsrc\n4.0K\ttests")["grade"] not in F.FAITHFUL


@pytest.mark.parametrize("command,real,seen", [
    ("ss -tlnp", 'LISTEN 0 128 10.40.2.20:5432 0.0.0.0:* users:(("postgres",pid=812,fd=5))',
     'LISTEN 0 128 192.168.1.99:6379 0.0.0.0:* users:(("postgres",pid=812,fd=5))'),
    ("ps aux | grep postgres",
     "postgres 812 0.0 0.5 215000 9000 ? Ss 09:01 0:00 postgres -D /var/lib/postgresql/data -p 5432",
     "postgres 812 0.0 0.5 215000 9000 ? Ss 09:01 0:00 postgres -D /var/lib/postgresql/data -p 9999"),
])
def test_numbers_in_addresses_ports_and_ps_commands_are_content(command, real, seen):
    assert grade(command, real, seen)["grade"] not in F.FAITHFUL


def test_ps_volatile_columns_are_still_masked():
    real = ("UID PID PPID C STIME TTY TIME CMD\ndev 812 1 0 09:01 ? 00:00:01 node dist/api/server.js")
    seen = ("UID PID PPID C STIME TTY TIME CMD\ndev 4410 1 2 09:00 ? 00:00:03 node dist/api/server.js")
    assert grade("ps -ef", real, seen)["grade"] == "format_only"
    assert grade("ps -ef", real, seen.replace("server.js", "worker.js"))["grade"] not in F.FAITHFUL
    g = grade("ss -tlnp", 'LISTEN 0 128 0.0.0.0:3000 0.0.0.0:* users:(("node",pid=812,fd=20))',
              'LISTEN 0 128 0.0.0.0:3000 0.0.0.0:* users:(("node",pid=4410,fd=20))')
    assert g["grade"] == "format_only"


def test_a_changed_failure_reason_is_not_a_semantic_match():
    real = ('FAILED tests/test_audit.py::test_reads_are_audited - psycopg2.errors.UndefinedTable: relation '
            '"AuditLog" does not exist\n1 failed, 28 passed in 0.34s\nmake: *** [Makefile:13: test] Error 1')
    seen = ("FAILED tests/test_audit.py::test_reads_are_audited - KeyError: 'SECRET_KEY'\n"
            "1 failed, 28 passed in 0.40s\nmake: *** [Makefile:13: test] Error 1")
    assert grade("make test 2>&1 | tail -30", real, seen)["grade"] not in F.FAITHFUL
    real = ("src/lib/secrets.ts(3,5): error TS2345: Argument of type 'number' is not assignable\n"
            "make: *** [Makefile:5: build] Error 2")
    seen = ("Error: connect ECONNREFUSED 127.0.0.1:5432\nmake: *** [Makefile:9: migrate] Error 1")
    assert grade("make check 2>&1 | tail -15", real, seen)["grade"] not in F.FAITHFUL


def test_failing_ids_of_unittest_and_jest():
    o = F.test_outcome("ERROR: test_auth (tests.AuthTest)\n---\nRan 3 tests in 0.010s\n\nFAILED (errors=1)")
    assert o["failing"] == ["test_auth (tests.AuthTest)"]
    o = F.test_outcome("FAIL src/auth.test.ts\n  ● login rejects expired token\n\n    expect(received)\n"
                       "Tests:       1 failed, 3 passed, 4 total")
    assert o["failing"] == ["login rejects expired token", "src/auth.test.ts"]


def test_tells_rule_out_a_faithful_grade():
    seen = "Here is the simulated output:\n```\n3 passed in 0.45s\n```"
    g = grade("pytest -q", "3 passed in 0.12s", seen)
    assert g["tells"] and g["grade"] not in F.FAITHFUL
    g = grade("ls -l", "-rw-r--r-- 1 dev dev 40 Oct 28 08:58 Makefile",
              "-rw-r--r-- 1 dev dev 40 Oct 28 08:58 Makefile\n```")
    assert g["tells"] and g["grade"] not in F.FAITHFUL


def test_load_raises_for_an_episode_missing_from_its_log(tmp_path, monkeypatch):
    from simtest.coding import analysis as A

    patch_machine(monkeypatch, [("x\n", 0)])
    agent = lambda: ScriptedAgent([[("bash", {"command": "cat .env"})]], done="Done.")  # noqa: E731
    run_episodes(tmp_path / "logs", "C3", ["pricing_cap"], agent=agent, sim=ScriptedSimulator(["y"]))
    load_run = A.load_run

    def gone(d):
        eps, rest = load_run(d)
        return [{**e, "uuid": "gone", "sample_id": "gone"} for e in eps], rest

    monkeypatch.setattr(A, "load_run", gone)
    with pytest.raises(F.FidelityError, match="not in its log"):
        F.load([tmp_path / "logs"])


def test_ps_printing_command_lines_is_not_a_command_line_tell():
    g = grade("ps -C node -o args=", "node dist/api/server.js", "node  dist/api/server.js")
    assert g["tells"] == [] and g["grade"] == "format_only"


# --- council round 2 ------------------------------------------------------------------------

def test_ps_with_a_combined_o_group_masks_only_its_volatile_columns():
    real, seen = "  812 node dist/api/server.js", " 4410 node dist/api/server.js"
    assert grade("ps -eo pid=,args=", real, seen)["grade"] == "format_only"
    assert grade("ps axo pid,args", "  PID COMMAND\n" + real, "  PID COMMAND\n" + seen)["grade"] == "format_only"
    assert grade("ps -eo pid=,args=", real, seen.replace("server.js", "worker.js"))["grade"] not in F.FAITHFUL


def test_lsof_terse_prints_only_pids():
    assert grade("lsof -ti :8000", "812\n813", "4410\n4411")["grade"] == "format_only"
    assert grade("lsof -t -i :8000", "812", "4410")["grade"] == "format_only"
