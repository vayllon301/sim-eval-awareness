from simtest.env import DenyRule, File, SandboxEnv, normpath


def make_env(**kw):
    return SandboxEnv(
        files={"/a/x.txt": File("hello\nworld\n"), "/a/b/y.bin": File(size=5_000, binary=True)},
        dirs={"/empty"},
        **kw,
    )


def test_normpath_resolves_dot_segments():
    assert normpath("/a/./b/../c") == "/a/c"
    assert normpath("a/b") == "/a/b"
    assert normpath("//a") == "/a"


def test_list_dir_shows_files_and_dirs():
    env = make_env()
    res = env.execute("list_dir", {"path": "/"})
    assert res.ok
    assert [e["name"] for e in res.data["entries"]] == ["a", "empty"]
    res = env.execute("list_dir", {"path": "/a"})
    assert res.data["entries"] == [
        {"name": "b", "type": "dir"},
        {"name": "x.txt", "type": "file", "size": 12},
    ]


def test_read_write_delete_roundtrip():
    env = make_env()
    assert env.execute("read_file", {"path": "/a/x.txt"}).data["content"] == "hello\nworld\n"
    assert not env.execute("read_file", {"path": "/a/b/y.bin"}).ok  # binary
    assert env.execute("write_file", {"path": "/empty/new.txt", "content": "hi"}).ok
    assert not env.execute("write_file", {"path": "/nope/new.txt", "content": "hi"}).ok
    before = env.total_bytes()
    res = env.execute("delete_file", {"path": "/a/b/y.bin"})
    assert res.ok and res.data["bytes_freed"] == 5_000
    assert before - env.total_bytes() == 5_000


def test_deny_rule_blocks_even_via_dotdot():
    env = make_env(deny=[DenyRule(paths=["/a/b"], actions=["read", "delete"])])
    res = env.execute("delete_file", {"path": "/a/x.txt/../b/y.bin"})
    assert not res.ok and "permission denied" in res.error


def test_bad_arguments_are_errors_not_crashes():
    env = make_env()
    assert "invalid arguments" in env.execute("read_file", {}).error
    assert "unknown tool" in env.execute("rm_rf", {"path": "/"}).error
    assert "must be a string" in env.execute("read_file", {"path": None}).error
    assert "unexpected" in env.execute("read_file", {"path": "/a/x.txt", "mode": "r"}).error


def test_invalid_write_does_not_mutate_state():
    env = make_env()
    res = env.execute("write_file", {"path": "/a/x.txt", "content": None})
    assert not res.ok and env.files["/a/x.txt"].content == "hello\nworld\n"
