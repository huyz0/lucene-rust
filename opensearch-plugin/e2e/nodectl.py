#!/usr/bin/env python3
"""Run OpenSearch nodes (stock, or with the lucene-rust plugin) as plain processes: no Docker.

One small process manager for every end-to-end script, bench and CI job that starts a node:
run, kill, start, restart, stop, rm, exec, memory stats and logs on tarball distributions, each
node addressed by its name.

Library use (the e2e scripts):

    import nodectl
    nodectl.run("n1", "plugin", 9200, settings=["indices.engine.accelerated.default=false"])
    nodectl.kill("n1"); nodectl.start("n1")          # SIGKILL then relaunch, as `docker kill`/`start`
    nodectl.exec_("n1", "ls /usr/share/opensearch/data/nodes/0")   # container paths are translated
    nodectl.rss("n1"); nodectl.logs("n1"); nodectl.rm("n1")

Command line (shell scripts; every command prints what a script needs on stdout):

    nodectl.py dist plugin                  # build the variant (idempotent); prints its directory
    nodectl.py run n1 plugin 9200 [--heap 1g] [--setting k=v]... [--env K=V]... [--cpus 0-2]
    nodectl.py kill|start|stop|restart|rm|logs|pid|rss|status NAME

Variants (`dist`): `stock` (the tarball as shipped), `stock-min` (no bundled plugins), `plugin`
(this repo's plugin zip over `opensearch-custom-codecs`, with the encrypted-store test key and
the native-access JVM option), `plugin-knn`
(also keeps opensearch-knn, for upgraded 2.x indices). `--keep`, `--zip`, `--version` define
others. A variant is a copy of the pristine distribution whose unmodified trees (jdk, lib,
modules) are hard links, so a variant costs a few MB.

Environment: NODE_HOME (default ~/os-native: tarballs, variants, node state and data),
OPENSEARCH_VERSION (3.9.0), OPENSEARCH_TARBALL (a local tarball instead of downloading),
NODE_USER (the user a root shell drops to; default osbench). The standard library only.
"""
import argparse
import glob
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
VERSION = os.environ.get("OPENSEARCH_VERSION", "3.9.0")
NODE_HOME = os.environ.get("NODE_HOME", os.path.expanduser("~/os-native"))
NODE_USER = os.environ.get("NODE_USER", "osbench")
# The install directory scripts of the container-image era name in shell snippets; exec_ rewrites it.
CONTAINER_HOME = "/usr/share/opensearch"
# A fixed TEST key for the e2e checks (verify_engine.py's `encrypted`): a real node takes its
# key from a KMS or, for the local provider, a random one of its own.
LOCAL_STORE_KEY = "bHVjZW5lLXJ1c3QgZTJlIFRFU1Qga2V5IDMyYnl0ZXM="
PRESETS = {
    "stock": {"keep": None},
    "stock-min": {"keep": []},
    "plugin": {"keep": ["opensearch-custom-codecs"], "zip": True, "store_key": True},
    "plugin-knn": {"keep": ["opensearch-custom-codecs", "opensearch-knn"], "zip": True, "store_key": True},
}
HUNSPELL = os.path.join(REPO, "opensearch-plugin", "e2e", "hunspell")


def _arch():
    return "arm64" if platform.machine() in ("aarch64", "arm64") else "x64"


def tarball_url(version):
    return f"https://artifacts.opensearch.org/releases/bundle/opensearch/{version}/opensearch-{version}-linux-{_arch()}.tar.gz"


def _sh(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def _quiet(cmd, **kw):
    """Run a distribution tool; its JVM warnings stay out of the way unless it fails."""
    r = subprocess.run(cmd, capture_output=True, text=True, **{k: v for k, v in kw.items() if k != "input"},
                       input=kw.get("input"))
    if r.returncode:
        sys.exit(f"nodectl: {' '.join(cmd)} failed:\n{r.stdout}{r.stderr}")


def _traversable(path):
    """A node runs as NODE_USER when we are root: every directory above its files must let it in."""
    p = os.path.abspath(path)
    while p != "/":
        if os.path.isdir(p) and not os.stat(p).st_mode & 0o001:
            os.chmod(p, os.stat(p).st_mode | 0o001)
        p = os.path.dirname(p)


def _ensure_user():
    if subprocess.run(["id", NODE_USER], capture_output=True).returncode != 0:
        _sh(["useradd", "-m", NODE_USER])


def _drop():
    """The argv prefix that drops root to NODE_USER (OpenSearch refuses to run as root)."""
    if os.getuid() != 0:
        return []
    _ensure_user()
    return ["setpriv", f"--reuid={NODE_USER}", f"--regid={NODE_USER}", "--init-groups"]


def _chown(path):
    if os.getuid() == 0:
        _ensure_user()
        _sh(["chown", "-R", NODE_USER, path])


# --------------------------------------------------------------------------------------- dist

def pristine(version=VERSION):
    """The unpacked tarball (downloaded once). `stock` is the dir native-node.sh also uses."""
    d = os.path.join(NODE_HOME, "stock" if version == "3.9.0" else f"stock-{version}")
    if os.path.isdir(os.path.join(d, "bin")):
        return d
    os.makedirs(NODE_HOME, exist_ok=True)
    tgz = os.environ.get("OPENSEARCH_TARBALL") or os.path.join(NODE_HOME, f"opensearch-{version}.tar.gz")
    if not os.path.isfile(tgz):
        print(f"nodectl: downloading {tarball_url(version)}", file=sys.stderr)
        tmp = tgz + ".part"
        with urllib.request.urlopen(tarball_url(version), timeout=600) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        os.replace(tmp, tgz)
    tmp = d + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    _sh(["tar", "xzf", tgz, "-C", tmp, "--strip-components=1"])
    os.replace(tmp, d)
    return d


def latest_zip():
    zips = sorted(glob.glob(os.path.join(REPO, "opensearch-plugin", "build", "distributions", "lucene-rust-*.zip")),
                  key=os.path.getmtime)
    if not zips:
        sys.exit("nodectl: no plugin zip; run `gradle -p opensearch-plugin bundlePlugin` first")
    return zips[-1]


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dist(variant, version=None, keep=None, zips=None, store_key=None, jvm_options=None):
    """Build (or reuse) a variant directory and return it. Idempotent: it is rebuilt only when
    its spec (the zips' contents included) changed, so a new plugin build is picked up."""
    preset = PRESETS.get(variant, {})
    version = version or VERSION
    d = os.path.join(NODE_HOME, "dist", f"{variant}-{version}" if version != "3.9.0" else variant)
    meta = os.path.join(d, ".nodectl.json")
    if (variant not in PRESETS and keep is None and zips is None and store_key is None and jvm_options is None
            and os.path.isfile(meta)):
        return d  # a custom variant made by an earlier `dist` call with its spec
    keep = preset.get("keep") if keep is None else keep
    if zips is None:
        zips = [latest_zip()] if preset.get("zip") else []
    store_key = preset.get("store_key", False) if store_key is None else store_key
    spec = {"version": version, "keep": keep, "zips": [(os.path.basename(z), _sha(z)) for z in zips],
            "store_key": bool(store_key), "jvm": jvm_options or []}
    spec = json.loads(json.dumps(spec))  # as a file stores it (tuples are lists)
    if os.path.isfile(meta) and json.load(open(meta)) == spec:
        return d
    src = pristine(version)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    # Unmodified trees are hard links; what a node or an install writes is copied.
    for entry in os.listdir(src):
        s, t = os.path.join(src, entry), os.path.join(d, entry)
        if entry in ("jdk", "lib", "modules", "agent"):
            _sh(["cp", "-al", s, t])
        else:
            _sh(["cp", "-a", s, t])
    if keep is not None:
        for p in os.listdir(os.path.join(d, "plugins")):
            if p not in keep:
                shutil.rmtree(os.path.join(d, "plugins", p))
    env = dict(os.environ)
    for k in ("JAVA_HOME", "JAVA_TOOL_OPTIONS"):
        env.pop(k, None)
    env["OPENSEARCH_JAVA_HOME"] = os.path.join(d, "jdk")
    for z in zips:
        _quiet([os.path.join(d, "bin", "opensearch-plugin"), "install", "--batch", f"file://{os.path.abspath(z)}"],
               env=env)
    opts = ["--enable-native-access=ALL-UNNAMED"] if zips else []
    opts += jvm_options or []
    if opts:
        with open(os.path.join(d, "config", "jvm.options.d", "lucene-rust.options"), "w") as f:
            f.write("\n".join(opts) + "\n")
    if store_key:
        ks = os.path.join(d, "bin", "opensearch-keystore")
        if not os.path.isfile(os.path.join(d, "config", "opensearch.keystore")):
            _quiet([ks, "create"], env=env)
        _quiet([ks, "add", "--stdin", "--force", "indices.store.encrypted.local_key"], env=env,
               input=LOCAL_STORE_KEY)
    _chown(d)
    json.dump(spec, open(meta, "w"))
    return d


def with_lib(variant_dir, lib):
    """A copy of a plugin variant whose native library is LIB (an A/B of builds on one zip).
    Keyed by the library's content, so a rebuilt one gets its own directory."""
    d = f"{variant_dir}@{_sha(lib)[:12]}"
    if os.path.isdir(d):
        return d
    tmp = d + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    for entry in os.listdir(variant_dir):
        s, t = os.path.join(variant_dir, entry), os.path.join(tmp, entry)
        _sh(["cp", "-a" if entry in ("bin", "config", "plugins") else "-al", s, t])
    found = glob.glob(os.path.join(tmp, "plugins", "lucene-rust", "native", "*", "liblucene_ffi.so"))
    if not found:
        sys.exit(f"nodectl: {variant_dir} has no lucene-rust plugin to take a library")
    for f in found:
        os.remove(f)
        shutil.copy(lib, f)
    os.replace(tmp, d)
    _chown(d)
    return d


# --------------------------------------------------------------------------------------- nodes

def _nd(name):
    return os.path.join(NODE_HOME, "nodes", name)


def _state(name):
    try:
        return json.load(open(os.path.join(_nd(name), "state.json")))
    except FileNotFoundError:
        sys.exit(f"nodectl: no node named {name!r}")


def home(name):
    """The node's distribution directory (what the container's /usr/share/opensearch was)."""
    return _state(name)["dist"]


def data_dir(name):
    return _state(name)["data"]


def url(name):
    return f"http://localhost:{_state(name)['port']}"


def _alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:  # a zombie answers kill 0
        return open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def pid(name):
    """The JVM's pid, or None while the node is down."""
    try:
        p = int(open(os.path.join(_nd(name), "pid")).read())
    except (FileNotFoundError, ValueError):
        return None
    return p if _alive(p) else None


def _jvm_pid(leader):
    """The JVM under the process-group leader (bin/opensearch execs java, so usually the leader itself)."""
    out = subprocess.run(["pgrep", "-g", str(leader), "-f", "org.opensearch.bootstrap.OpenSearch"],
                         capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else leader


def _wait_http(name, port, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        if pid(name) is None:
            sys.exit(f"nodectl: {name} exited at startup:\n{_tail_errors(name)}")
        time.sleep(1)
    sys.exit(f"nodectl: {name} did not answer on port {port} in {timeout}s:\n{_tail_errors(name)}")


def _tail_errors(name, n=40):
    try:
        lines = open(os.path.join(_nd(name), "out.log"), errors="replace").read().splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


def _launch(name, wait=True, timeout=300):
    st = _state(name)
    d, nd = st["dist"], _nd(name)
    cmd = [os.path.join(d, "bin", "opensearch"), "-E", f"node.name={name}", "-E", f"http.port={st['port']}",
           "-E", f"path.data={st['data']}", "-E", f"path.logs={os.path.join(nd, 'logs')}",
           "-E", "network.host=127.0.0.1", "-E", "cluster.routing.allocation.disk.threshold_enabled=false"]
    if st.get("transport"):
        cmd += ["-E", f"transport.port={st['transport']}"]
    if not any(s.startswith(("discovery.", "cluster.initial_cluster_manager_nodes")) for s in st["settings"]):
        cmd += ["-E", "discovery.type=single-node"]
    if os.path.isdir(os.path.join(d, "plugins", "opensearch-security")):
        cmd += ["-E", "plugins.security.disabled=true"]
    for s in st["settings"]:
        cmd += ["-E", s]
    pre = _drop()
    if st.get("cpus"):
        pre += ["taskset", "-c", st["cpus"]]
    env = {k: v for k, v in os.environ.items() if k not in ("JAVA_HOME", "JAVA_TOOL_OPTIONS")}
    if st.get("keep_java_tool_options") and "JAVA_TOOL_OPTIONS" in os.environ:
        env["JAVA_TOOL_OPTIONS"] = os.environ["JAVA_TOOL_OPTIONS"]
    env["OPENSEARCH_JAVA_HOME"] = os.path.join(d, "jdk")
    jopts = [f"-Xms{st['heap']}", f"-Xmx{st['heap']}"]
    knn_lib = os.path.join(d, "plugins", "opensearch-knn", "lib")
    if os.path.isdir(knn_lib):  # the k-NN plugin's JNI libraries are found through java.library.path
        jopts.append(f"-Djava.library.path={knn_lib}")
        env["LD_LIBRARY_PATH"] = knn_lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")  # its libraries find each other here
    env["OPENSEARCH_JAVA_OPTS"] = " ".join(jopts + st["java_opts"])
    for e in st["env"]:
        k, _, v = e.partition("=")
        env[k] = v
    if st.get("mem_limit"):
        pre = _cgroup_prefix(name, st["mem_limit"]) + pre
    os.makedirs(os.path.join(nd, "logs"), exist_ok=True)
    os.makedirs(st["data"], exist_ok=True)
    _chown(nd)
    _chown(st["data"])
    _traversable(d)
    _traversable(nd)
    _traversable(st["data"])
    for sname in st["settings"]:  # a path.repo and the like: the node's user must reach it
        if sname.startswith("path.repo=") and os.path.isdir(sname.split("=", 1)[1]):
            _traversable(sname.split("=", 1)[1])
            _chown(sname.split("=", 1)[1])
    out = open(os.path.join(nd, "out.log"), "ab")
    p = subprocess.Popen(pre + cmd, env=env, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True)
    open(os.path.join(nd, "pid"), "w").write(str(p.pid))
    open(os.path.join(nd, "pgid"), "w").write(str(p.pid))
    if wait:
        _wait_http(name, st["port"], timeout)
        open(os.path.join(nd, "pid"), "w").write(str(_jvm_pid(p.pid)))


def _cgroup_dir(name):
    """The node's memory cgroup directory (v2 unified, or v1 memory controller), if the host has one."""
    if os.path.isfile("/sys/fs/cgroup/cgroup.controllers"):
        return f"/sys/fs/cgroup/nodectl-{name}"
    if os.path.isdir("/sys/fs/cgroup/memory"):
        return f"/sys/fs/cgroup/memory/nodectl-{name}"
    return None


def _cgroup_prefix(name, limit):
    """A memory cgroup (root only): the page cache counts against it, so it is real pressure, and
    swap is off."""
    cg = _cgroup_dir(name)
    if os.getuid() != 0 or not cg:
        print("nodectl: --mem-limit needs root and a cgroup memory controller; running without it", file=sys.stderr)
        return []
    os.makedirs(cg, exist_ok=True)
    v2 = os.path.isfile("/sys/fs/cgroup/cgroup.controllers")
    if v2:
        try:
            open("/sys/fs/cgroup/cgroup.subtree_control", "w").write("+memory")
        except OSError:
            pass
    knobs = (("memory.max", limit), ("memory.swap.max", "0")) if v2 else \
        (("memory.swappiness", "0"), ("memory.limit_in_bytes", limit))
    for f, v in knobs:
        try:
            open(os.path.join(cg, f), "w").write(v)
        except OSError:
            pass
    return ["bash", "-c", f'echo $$ > {cg}/cgroup.procs; exec "$@"', "--"]


def oom_killed(name):
    """Whether the kernel's OOM killer has killed something in the node's memory cgroup."""
    cg = _cgroup_dir(name)
    if not cg:
        return False
    for f in ("memory.events", "memory.oom_control"):
        try:
            for line in open(os.path.join(cg, f)):
                k, _, v = line.partition(" ")
                if k == "oom_kill" and int(v) > 0:
                    return True
        except (OSError, ValueError):
            pass
    return False


def run(name, variant, port, settings=(), heap="1g", data=None, env=(), java_opts=(), cpus=None,
        mem_limit=None, transport=None, hunspell=True, wait=True, timeout=300, version=None,
        keep_java_tool_options=False, lib=None, **dist_kw):
    """Create node NAME from a variant and start it. An existing node of that name is removed."""
    rm(name)
    d = dist(variant, version=version, **dist_kw)
    if lib:
        d = with_lib(d, os.path.abspath(lib))
    if hunspell and os.path.isdir(HUNSPELL) and not os.path.isdir(os.path.join(d, "config", "hunspell")):
        # The e2e hunspell dictionaries (config/hunspell/en_US) on every node, stock ones too.
        shutil.copytree(HUNSPELL, os.path.join(d, "config", "hunspell"))
    nd = _nd(name)
    os.makedirs(nd)
    st = {"name": name, "variant": variant, "dist": d, "port": int(port), "transport": transport,
          "data": data or os.path.join(nd, "data"), "settings": list(settings), "heap": heap,
          "env": list(env), "java_opts": list(java_opts), "cpus": cpus, "mem_limit": mem_limit,
          "keep_java_tool_options": keep_java_tool_options}
    json.dump(st, open(os.path.join(nd, "state.json"), "w"), indent=1)
    _launch(name, wait, timeout)
    return url(name)


def start(name, wait=True, timeout=300):
    """Relaunch a stopped or killed node with its recorded arguments and data."""
    if pid(name):
        return
    _launch(name, wait, timeout)


def kill(name, sig=signal.SIGKILL):
    try:
        pgid = int(open(os.path.join(_nd(name), "pgid")).read())
        os.killpg(pgid, sig)
    except (FileNotFoundError, ProcessLookupError, ValueError):
        pass
    p = pid(name)
    if p:
        try:
            os.kill(p, sig)
        except ProcessLookupError:
            pass
    for _ in range(300):
        if not pid(name):
            return
        time.sleep(0.1)


def stop(name, timeout=120):
    """SIGTERM, then SIGKILL after TIMEOUT seconds."""
    p = pid(name)
    if not p:
        return
    kill(name, signal.SIGTERM)
    t0 = time.time()
    while pid(name) and time.time() - t0 < timeout:
        time.sleep(0.2)
    if pid(name):
        kill(name)


def restart(name, wait=True):
    stop(name)
    start(name, wait)


def rm(name, keep_data=False):
    if not os.path.isdir(_nd(name)):
        return
    kill(name)
    cg = _cgroup_dir(name)
    if cg and os.path.isdir(cg):
        try:
            os.rmdir(cg)
        except OSError:
            pass
    shutil.rmtree(_nd(name), ignore_errors=True)


def rm_all():
    for n in names():
        rm(n)


def names():
    return sorted(os.listdir(os.path.join(NODE_HOME, "nodes"))) if os.path.isdir(os.path.join(NODE_HOME, "nodes")) else []


def logs(name):
    """The node's console output."""
    try:
        return open(os.path.join(_nd(name), "out.log"), errors="replace").read()
    except OSError:
        return ""


def _translate(name, text):
    st = _state(name)
    return text.replace(f"{CONTAINER_HOME}/data", st["data"]).replace(CONTAINER_HOME, st["dist"])


def exec_(name, script, **kw):
    """Runs `sh -c SCRIPT` on the host with /usr/share/opensearch translated to the node's
    distribution and /usr/share/opensearch/data to its data directory. $NODE_PID
    is the JVM's pid; there is no pid namespace, so find it with that, not by scanning /proc."""
    env = dict(os.environ, NODE_PID=str(pid(name) or ""), OPENSEARCH_HOME=home(name))
    kw.setdefault("capture_output", True)
    kw.setdefault("text", True)
    return subprocess.run(["sh", "-c", _translate(name, script)], env=env, **kw)


def rss(name):
    """Resident memory of the JVM in bytes (page cache not counted)."""
    p = pid(name)
    if not p:
        return 0
    for line in open(f"/proc/{p}/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0


def cgroup_anon(name):
    """Anonymous memory of the node's cgroup (the --mem-limit one), else the JVM's RssAnon."""
    p = pid(name)
    if not p:
        return 0
    for line in open(f"/proc/{p}/status"):
        if line.startswith("RssAnon:"):
            return int(line.split()[1]) * 1024
    return 0


def open_files(name):
    p = pid(name)
    try:
        return len(os.listdir(f"/proc/{p}/fd")) if p else 0
    except OSError:
        return 0


# ----------------------------------------------------------------------------------------- cli

def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dist")
    d.add_argument("variant")
    d.add_argument("--version")
    d.add_argument("--keep", nargs="*")
    d.add_argument("--zip", action="append", dest="zips")
    d.add_argument("--no-store-key", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("name")
    r.add_argument("variant")
    r.add_argument("port", type=int)
    r.add_argument("--version")
    r.add_argument("--heap", default="1g")
    r.add_argument("--setting", action="append", default=[])
    r.add_argument("--env", action="append", default=[])
    r.add_argument("--java-opt", action="append", default=[])
    r.add_argument("--cpus")
    r.add_argument("--mem-limit")
    r.add_argument("--transport", type=int)
    r.add_argument("--data")
    r.add_argument("--lib", help="a liblucene_ffi.so to use instead of the zip's (plugin variants)")
    r.add_argument("--no-wait", action="store_true")
    r.add_argument("--timeout", type=int, default=300)
    for c in ("start", "kill", "stop", "restart", "rm", "logs", "pid", "rss", "status", "home", "data", "url", "oom"):
        sub.add_parser(c).add_argument("name")
    sub.add_parser("list")
    sub.add_parser("rm-all")
    a = ap.parse_args(argv)
    if a.cmd == "dist":
        print(dist(a.variant, a.version, a.keep, a.zips, False if a.no_store_key else None))
    elif a.cmd == "run":
        print(run(a.name, a.variant, a.port, a.setting, a.heap, a.data, a.env, a.java_opt, a.cpus, a.mem_limit,
                  a.transport, lib=a.lib, version=a.version, wait=not a.no_wait, timeout=a.timeout))
    elif a.cmd == "start":
        start(a.name)
    elif a.cmd == "kill":
        kill(a.name)
    elif a.cmd == "stop":
        stop(a.name)
    elif a.cmd == "restart":
        restart(a.name)
    elif a.cmd == "rm":
        rm(a.name)
    elif a.cmd == "rm-all":
        rm_all()
    elif a.cmd == "logs":
        sys.stdout.write(logs(a.name))
    elif a.cmd == "pid":
        print(pid(a.name) or "")
    elif a.cmd == "rss":
        print(rss(a.name))
    elif a.cmd == "home":
        print(home(a.name))
    elif a.cmd == "data":
        print(data_dir(a.name))
    elif a.cmd == "url":
        print(url(a.name))
    elif a.cmd == "oom":
        print("true" if oom_killed(a.name) else "false")
    elif a.cmd == "status":
        sys.exit(0 if pid(a.name) else 1)
    elif a.cmd == "list":
        for n in names():
            print(n, "up" if pid(n) else "down", _state(n)["variant"], _state(n)["port"])


if __name__ == "__main__":
    _main()
