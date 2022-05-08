import os
import toml
import spacy
import requests
import dill as pickle
from json import dumps
from numpy import cumsum
from queue import Queue
from shutil import rmtree
from itertools import chain
from zipfile import ZipFile
from time import time, sleep
from random import choice, seed
from collections import Counter
from nltk.corpus import stopwords
from mmap import mmap, ACCESS_READ
from rust_demangler import demangle
from argparse import ArgumentParser
from operator import __mul__ as mul
from re import compile, ASCII, DOTALL
from tempfile import NamedTemporaryFile
from functools import lru_cache, reduce
from magic import from_file as magic_info
from multiprocessing.sharedctypes import Value
from multiprocessing import Process, Lock as mLock
from subprocess import call, DEVNULL, TimeoutExpired
from threading import Thread, get_native_id, Lock as tLock
from github import Github
from github.GithubException import RateLimitExceededException

from extract_funcs import parse_json


nltk_stopwords = set(stopwords.words("english"))
custom_stop_words = {
    "return", "use", "value", "let", "mut", "type", "sync", "ex", "mc", "sm", "cm", "global",
    "tuple", "call", "crust", "pointer", "vr", "datum", "fill", "ref", "simple",
    "default", "byte", "buff", "null", "numb", "slice", "tx", "ptr", "ext", "content", "cairo",
    "update", "bite", "map", "int", "object", "obj", "type", "bindgen", "const", "constant",
    "one", "run", "change", "make", "op", "entry", "true", "end", "next", "group", "scale",
    "cmd", "point", "float", "apply", "control", "without", "generic", "template",
    "main", "method", "range", "command", "item", "bit", "empty", "current", "target", "fix",
    "offset", "zero", "result", "double", "visit", "option", "opt", "rx", "try", "internal",
    "prepare", "reference", "char", "execute", "exec", "emit", "power", "element", "local",
    "work", "raw", "api", "context", "ctx", "self", "proc", "ctrl", "last", "per", "need",
    "class", "struct", "two", "iter", "base", "impl", "ic", "non", "var", "single", "st",
    "implement", "variant", "\ufeff1", "require", "feature", "mk", "level", "enum", "long",
    "argument", "args", "arg", "must", "brief", "note", "example", "pass", "much", "none",
    "may", "structure", "foo", "uint", "integer", "bool", "hex", "dt", "app", "instr", "inst",
    "lead", "slot", "rt"
}
stop_words = nltk_stopwords | custom_stop_words

classes = {
    "destroy": [
        "clean", "cleanup", "clear", "remove", "drop", "reset", "free", "shutdown", "bite", "finish", "erase",
        "disable", "delete", "del", "destroy", "stop", "release", "leave", "exit", "sub", "unregister", "cancel",
        "suspend", "abort", "deinit", "finalize", "escape", "gc"
    ],
    "init": [
        "init", "initialize", "activate", "new", "register", "reg", "create", "gen", "generate", "start", "enable",
        "setup", "build", "begin", "resume", "construct", "install", "def", "define", "refresh", "fini", "decl"
    ],
    "get": [
        "get", "take", "pop", "select", "attr", "attribute", "field", "param", "parameter", "params",
        "property", "prop"
    ],
    "rrc": [
        "dissect", "lte", "rrc", "pdu", "nr", "cell", "nbap", "rnsap", "srsran"
    ],
    "log": [
        "log", "config", "cfg", "conf", "configure", "dump", "store", "commit", "save", "specify", "setting", "env",
        "report"
    ],
    "test": [
        "test", "detect", "layout", "bench", "validate", "check", "verify", "complete", "probe", "match", "valid",
        "control", "success", "fail", "invalid", "debug", "br", "trace"
    ],
    "database": [
        "sqlite", "query", "table", "db", "schema", "record", "transaction"
    ],
    "network": [
        "network", "net", "send", "snd", "recv", "rcv", "receive", "buf", "buff", "request", "req", "title",
        "message", "msg", "packet", "header", "hdr", "link", "channel", "page", "port", "fetch", "client", "connect",
        "server", "udp", "http", "ngx", "pdu", "response", "address", "addr", "ip", "bind", "attach", "detach",
        "socket", "sock", "host", "async", "session", "tcp", "ack", "post", "poll", "bridge", "proto", "service",
        "peer", "url", "ie", "tls", "ipv", "vlan", "load", "accept", "proxy", "connection", "conn", "disconnect",
        "reply", "body", "reflect", "ieee", "src", "dst", "protocol", "policy", "dispatch", "dns", "org", "web",
        "submit", "remote", "rpc", "grpc"
    ],
    "file": [
        "read", "reader", "write", "writer", "io", "file", "filename", "fd", "open", "close", "pipe", "seek", "doc",
        "desc", "descriptor"
    ],
    "string": [
        "parse", "parser", "str", "string", "text", "encode", "encoder", "enc", "decode", "pack", "unpack",
        "input", "line", "filter", "reg", "json", "xml", "pdf", "replace", "utf", "font", "split", "pattern"
    ],
    "format": [
        "convert", "to", "format", "fmt", "serialize", "deserialize", "transform", "normalize", "wrap", "align",
        "translate", "clip"
    ],
    "compress": [
        "compress", "compression", "compressor", "gzip", "zip", "bz", "gz", "tar", "xz", "archive", "lzma", "extract",
        "metadata"
    ],
    "error": [
        "error", "assert", "eq", "warn", "warning", "message", "msg", "err", "handle", "parse", "syntax", "trigger"
    ],
    "device": [
        "device", "dev", "dma", "dm", "channel", "ioctl", "driver", "drv", "pci", "pcie", "usb", "gpio", "phy", "mac",
        "intel", "mlxsw", "ib", "hal", "bus", "adc", "pcm", "tegra", "eth", "rte", "mesh", "dp", "spi", "sp",
        "amdgpu", "lpfc", "rtl", "sd", "arm", "pin", "fw", "omap", "flash", "mouse", "disk", "partition", "ixgbe",
        "ble", "bt", "native", "endpoint", "arch"
    ],
    "os": [
        "system", "sys", "os", "xfs", "nfs", "btrfs", "mount", "mkdir", "kernel", "engine", "inode",
        "core", "basic", "dfs"
    ],
    "parallel": [
        "thread", "process", "cpu", "hw", "spawn", "schedule", "scheduler", "task", "job", "batch", "worker",
        "master", "slave"
    ],
    "async": [
        "block", "lock", "mutex", "unlock", "async"
    ],
    "signal": [
        "handle", "signal", "interrupt", "handler", "irq", "intr", "hook", "cb", "callback", "action"
    ],
    "event": [
        "event", "state", "mode", "mod", "wait", "ready", "idle", "complete", "status", "code",
        "flag", "track", "notify", "atomic", "pend", "monitor", "handle"
    ],
    "time": [
        "time", "timer", "clock", "clk", "timeout", "wait", "rtc", "rte", "delay", "ms", "sec", "date", "sleep",
        "tick", "timestamp", "duration"
    ],
    "copy": [
        "copy", "cp", "move", "swap", "merge", "clone", "transfer"
    ],
    "path": [
        "path", "file", "dir", "directory", "root", "child", "parent"
    ],
    "graphic": [
        "show", "draw", "paint", "print", "output", "window", "win", "display", "frame", "flush", "video", "image",
        "stream", "render", "color", "gimp", "tool", "gl", "bound", "box", "gui", "view", "cursor", "button",
        "ui", "screen", "interface", "rgb", "menu", "texture", "pixel", "gfx", "widget", "shader", "ff", "wm",
        "style", "rect", "msa", "rotate", "angle", "degree", "deg", "snapshot", "scroll", "codec"
    ],
    "help": [
        "info", "information", "help", "helper", "lookup", "version", "support", "stats", "stat", "serial",
        "description"
    ],
    "matrix": [
        "matrix", "mm", "mul", "mask", "row", "col", "column", "vector", "vec", "height", "width"
    ],
    "container": [
        "queue", "list", "array", "vector", "vec", "cache", "heap", "fifo", "bitmap", "max", "min",
        "sum", "contain", "len", "length", "find", "search", "pattern", "match", "scan", "num", "size", "common",
        "position", "pos", "resize", "add", "insert", "put", "push", "append", "stack", "counter", "container", "count",
        "iterator", "iter", "reverse"
    ],
    "resource": [
        "mem", "memory", "ram", "space", "alloc", "allocate", "allocation", "free", "resource", "buf", "buffer",
        "chunk", "volume", "capacity", "limit", "pool", "storage", "usage", "available"
    ],
    "sort": [
        "sort", "order", "dissect", "loop", "seq", "skip", "sequence"
    ],
    "security": [
        "hash", "sign", "signature", "digest", "resolve", "crypto", "sha", "encrypt", "decrypt", "aes", "crypt", "md",
        "cipher", "cert", "ssl", "auth", "password", "secure", "key", "account", "login", "user", "acl",
        "profile", "allow", "private", "secret", "public", "access", "crc", "safe"
    ],
    "id": [
        "key", "id", "find", "index", "idx", "token", "tag"
    ],
    "eq": [
        "match", "eq", "equal", "compare", "cmp", "diff", "patch", "comp"
    ],
    "calc": [
        "compute", "calc", "calculate", "eval", "expression", "expr"
    ],
    "audio": [
        "audio", "capture", "volume", "codec", "hdmi", "sound", "snd", "radio", "vst", "midi", "track", "preset", "mic",
        "stream", "channel"
    ],
    "rand": [
        "random", "rand", "rng", "seed", "generate", "gen", "uniform", "sample"
    ],
    "library": [
        "export", "lib", "library", "symbol", "dyn", "fun", "fn", "func", "function", "import", "std",
        "dynamic", "module", "plugin", "package"
    ],
    "vm": [
        "vm", "kvm", "vcpu", "virtio", "virtual", "emulate", "kvmppc", "qemu"
    ],
    "graph": [
        "graph", "edge", "node", "vertex", "link", "neighbor", "tree", "child", "parent"
    ],
    "intrinsic": [
        "mm", "epi", "epu", "intr", "maskz", "anybitmemory", "xmmregister", "ymm", "avx"
    ],
    "speed":  [
        "speed", "perf", "rate", "fast", "velocity", "performance", "quick",
    ],
    "location": [
        "location", "distance", "position", "pos", "place", "coord", "region"
    ],
    "name": [
        "name", "rename", "alias"
    ],
    "tmp": [
        "temp", "tmp", "temporary"
    ],
    "assign": [
        "label", "assign", "mark", "cluster", "segment", "set"
    ],
    "term": [
        "term", "terminal", "tty", "sh", "shell", "cli", "tui", "console", "uart"
    ],
    "vcs": [
        "vcs", "svn", "git", "github", "branch", "repository", "repo", "merge", "commit"
    ],
    "bin": [
        "bin", "binary", "elf", "pe", "executable"
    ]
}


def extract_funcs_from_ll(crate_name, path_ll, name2label, name2body, path_dataset, randlen=5):
    one_from_names = rf"(?:{'|'.join(name2label.keys())})"
    str_pattern = rf"define internal %struct\.Memory\* @sub_[^_]+_(\w+?{one_from_names}[a-zA-Z0-9]{{20}})\(%struct\."
    str_pattern += r"State\* noalias nonnull align 1 %state, i64 %pc, %struct\.Memory\* noalias %memory\)[^}]+}\n"
    pattern = compile(bytes(str_pattern, encoding="utf-8"), ASCII | DOTALL)
    num_files = 0
    files = []

    with open(path_ll) as llfile:
        with mmap(llfile.fileno(), length=0, access=ACCESS_READ) as mmap_obj:
            for match in pattern.finditer(mmap_obj):
                fullname = demangle(str(match.group(1), encoding="utf8"))
                fullname = fullname[: fullname.rfind("::")]
                cratename = fullname[: fullname.find("::")]
                cratename = cratename[5: cratename.find("__", 5)] if cratename.startswith("__LT_") else cratename
                name = fullname[fullname.rfind("::") + 2:]
                if name in name2label and cratename == crate_name:
                    rand = "".join(choice([str(i) for i in range(10)]) for _ in range(randlen))
                    filename = os.path.join(os.path.join(path_dataset, name2label[name]), name + rand + ".ll")
                    with open(filename, "wb") as f, open(filename + ".info", "w") as finfo:
                        f.write(match.group(0))
                        finfo.write(f"{name2body[name]}")
                    num_files += 1
                    files.append(filename)
    return files


def predict_label(signatures, prob_path):
    func_names = []
    func_types = []
    for name, types, comments in (x for x in map(lambda l: l.strip().split("|"), signatures)):
        func_types.append(types.split(","))
        func_names.append(name)
    
    def tokenize_funcs(funcs):
        oneword = compile(r"^[a-z][a-z0-9]+|[A-Z][A-Z0-9]$")
        difCase = compile(r".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)")
        under_scores_split = compile(r"_")

        tokenized_funcs = []
        for f in funcs:
            if oneword.fullmatch(f):
                tokenized_funcs.append([f])
            elif "_" in f:
                tokenized_funcs.append([w for w in under_scores_split.split(f) if w])
            else:
                tokenized_funcs.append([w.group(0) for w in difCase.finditer(f) if w.group(0)])
        return tokenized_funcs

    tokenized_func_names = tokenize_funcs(func_names)

    def drop_wrong_symbols(tokenized_func_names):
        wrong_char = compile(r"[\d]")
        tokenized_func_names_ = []
        for tokenized_name in tokenized_func_names:
            processed_tokens = [wrong_char.sub("", token).lower() for token in tokenized_name if
                                wrong_char.sub("", token)]
            tokenized_func_names_.append(processed_tokens)

        return tokenized_func_names_

    tokenized_func_names = drop_wrong_symbols(tokenized_func_names)

    with open(prob_path, "rb") as f:
        prob = pickle.load(f)

    def split(word, start=1, end=20):
        return ((word[:i], word[i:]) for i in range(start, min(len(word) + 1, end)))

    @lru_cache(maxsize=10000)
    def segment(word, maxlen=500):
        if not word:
            return []
        if len(word) > maxlen:
            return segment(word[:maxlen]) + segment(word[maxlen:])
        candidates = ([first] + segment(remaining) for first, remaining in split(word))
        return max(candidates, key=lambda x: reduce(mul, map(prob, x), 1))

    def segmentize_corpus(tokenized_func_names, segmenter):
        tokenized_func_names = [list(chain(*(segmenter(token) for token in tokens)))
                                for tokens in tokenized_func_names]
        return tokenized_func_names

    tokenized_func_names = segmentize_corpus(tokenized_func_names, segment)

    def lemmatize_corpus(tokenized_func_names):
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        lengths = cumsum([0] + list(map(len, tokenized_func_names)))
        flat_tokens = list(chain(*tokenized_func_names))
        doc = spacy.tokens.Doc(nlp.vocab, words=flat_tokens)
        tokenized_func_names = [[token.lemma_ for token in doc[lengths[i - 1]: lengths[i]]]
                                for i in range(1, len(tokenized_func_names) + 1)]
        return tokenized_func_names

    tokenized_func_names = lemmatize_corpus(tokenized_func_names)

    def prune_vocabulary(func_names, tokenized_func_names, stop_words, max_ct=1e22, min_ct=1, min_len=2, max_len=20):
        word2count = Counter(chain(*tokenized_func_names))
        word2count = {k: v for k, v in sorted(word2count.items(), key=lambda item: item[1], reverse=True)
                      if min_ct <= v <= max_ct and min_len <= len(k) <= max_len and k not in stop_words}
        tokenized_func_names_ = []
        func_names_ = []
        for i, (name, tokenized_name) in enumerate(zip(func_names, tokenized_func_names)):
            processed_tokens = [token for token in tokenized_name if token in word2count]
            if processed_tokens:
                tokenized_func_names_.append(processed_tokens)
                func_names_.append(name)
                
        return func_names_, tokenized_func_names_

    func_names, tokenized_func_names = prune_vocabulary(func_names, tokenized_func_names, stop_words)

    def match(k, l):
        return k >= 0.5 * l

    def classify(func_names, tokenized_func_names, classes, class2funcs):
        tokenized_func_names_ = []
        func_names_ = []
        for name, tokenized_name in zip(func_names, tokenized_func_names):
            scores = {c: 0 for c in classes}
            for c, cl in classes.items():
                for tok in tokenized_name:
                    if tok in cl:
                        scores[c] += 1

            scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
            candidates = []
            cand_score = 0
            for c, sc in scores.items():
                is_match = match(sc, len(tokenized_name))
                if is_match and len(candidates) == 0:
                    cand_score = sc
                    candidates.append(c)
                else:
                    if is_match and sc == cand_score:
                        candidates.append(c)
                    else:
                        break

            if len(candidates) == 0:
                tokenized_func_names_.append(tokenized_name)
                func_names_.append(name)
            else:
                c = choice(candidates)
                class2funcs[c].append((name, tokenized_name))

        return func_names_, tokenized_func_names_

    class2funcs = {c: [] for c in classes}
    classify(func_names, tokenized_func_names, classes, class2funcs)

    return {n[0]: l for l, funcs in class2funcs.items() for n in funcs}


def lift(path_bin, path_get_cfg, path_ida, path_llvm_dis, mcsema_disas_timeout, llvm_version, os_version,
         path_mcsema_lift, max_bin_size=int(1000e6)):
    if os.stat(path_bin).st_size > max_bin_size:
        return
    path_cfg, path_bc, path_ll = f"{path_bin}.cfg", f"{path_bin}.bc", f"{path_bin}.ll"
    try:
        ret_code = call(f"exec wine {path_ida} -B -S\"{path_get_cfg} --output {path_cfg} --arch amd64 --os linux "
                        f"--entrypoint main\" {path_bin}", shell=True, timeout=mcsema_disas_timeout, stdout=DEVNULL,
                        stderr=DEVNULL)
        if ret_code or not os.stat(path_cfg).st_size:
            return
    except (TimeoutExpired, FileNotFoundError):
        return
    if path_mcsema_lift:
        cmd = f"{path_mcsema_lift} --output {path_bc} --arch amd64 --os linux --cfg {path_cfg}",
    else:
        cmd = f"docker run -it --rm --ipc=host -v {os.path.dirname(path_bin)}:/mcsema/local " \
              f"ghcr.io/lifting-bits/mcsema/mcsema-llvm{llvm_version}-{os_version}-amd64 --arch amd64 --os linux " \
              f"--cfg /mcsema/local/{os.path.basename(path_cfg)} --output /mcsema/local/{os.path.basename(path_bc)}"
    if call(cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL):
        return
    if call(f"{path_llvm_dis} {path_bc} -o {path_ll}", shell=True):
        return
    return path_ll if os.path.exists(path_ll) and os.stat(path_ll).st_size else None


def process_project(path, args):
    source_paths = []
    q = Queue()
    for p in os.scandir(path):
        q.put(p)
    while not q.empty():
        cur = q.get()
        if cur.is_file() and cur.name.endswith(".rs"):
            source_paths.append(os.path.abspath(cur))
        elif cur.is_dir():
            for p in os.scandir(cur):
                q.put(p)

    signatures = set()
    name2body = {}
    with NamedTemporaryFile("r+") as tmpfile:
        for p in source_paths:
            tmpfile.truncate(0)
            if call(f"{args['extractor_path']} {p} {tmpfile.name}", shell=True, stdout=DEVNULL, stderr=DEVNULL):
                continue
            tmpfile.seek(0)

            res = parse_json(tmpfile, append_sources=True)
            if not res:
                continue
            else:
                signatures |= set(res[0])
                name2body.update(res[1])
    name2label = predict_label(signatures, args["prob_path"])
    num_files = 0
    if not name2label:
        return num_files, None, None

    cargo_path = os.path.join(path, "Cargo.toml")
    ret_code = call(f"cargo build --manifest-path {cargo_path}", shell=True, stdout=DEVNULL, stderr=DEVNULL)

    bin2funcs = {}
    if ret_code == 0:
        with open(cargo_path) as f:
            content = toml.load(f)
        try:
            crate_name = content["package"]["name"]
        except KeyError:
            crate_name = None

        binary_paths = []
        for f in os.scandir(os.path.join(os.path.join(path, "target"), "debug")):
            if f.is_file():
                file_info = magic_info(os.path.abspath(f))
                if file_info == "current ar archive" or file_info.startswith("ELF"):
                    binary_paths.append(os.path.abspath(f))

        significant_binary_paths = []
        for p in binary_paths:
            crate_name = os.path.basename(p) if not crate_name else crate_name 
            path_ll = lift(p, args["get_cfg_path"], args["ida_path"], args["llvm_dis_path"],
                           args["mcsema_disas_timeout"], args["llvm_version"], args["os_version"],
                           args["mcsema_lift_path"])
            if path_ll:
                significant_binary_paths.append(p)
                files = extract_funcs_from_ll(crate_name, path_ll, name2label, name2body, args["dataset_ll_path"])
                num_files += len(files)
                bin2funcs[p] = files
    return num_files, source_paths, bin2funcs


def safe_call(p_lock, t_lock, f, *args, **kwargs):
    with p_lock, t_lock:
        res = f(*args, **kwargs)
    return res


def get_repo(idx, stars, prev_idx, g, repositories):
    # Search Api Limits, use fixed totalCount because PyGithub bug
    pagesize = 30
    totalCount = 1000
    dt = 5
    if idx.value == totalCount - 1:
        stars.value = repositories[totalCount - 1].stargazers_count
        idx.value = -1
    if prev_idx >= idx.value:
        repositories = g.search_repositories(query=f"stars:<={stars.value} language:rust", sort="stars", order="desc")
        prev_idx = -1

    idx.value += 1
    num_requests = idx.value // pagesize - prev_idx // pagesize
    remaining, _ = g.rate_limiting
    if remaining < num_requests:
        sleep(abs(g.rate_limiting_resettime - time()) + dt)
    try:
        repo = repositories[idx.value]
    except (RateLimitExceededException, requests.exceptions.ConnectTimeout):
        # use this exception because PyGithub bug rate_limiting calculation
        sleep(abs(g.rate_limiting_resettime - time()) + dt)
        repo = repositories[idx.value]

    return idx.value, repo, repositories


def remove_reduntant_files(path, significant_paths):
    q = Queue()
    for p in os.scandir(path):
        q.put(p)
    to_remove = []
    while not q.empty():
        f = q.get()
        if f.is_file():
            if os.path.abspath(f) in significant_paths:
                continue
            else:
                to_remove.append(f)
        else:
            for p in os.scandir(f):
                q.put(p)
    for f in to_remove:
        os.remove(f)


def gen_dataset(p_lock, t_lock, num_files, idx, stars, args, max_retries=3, time_to_wait=5):
    g = Github(args["token"])
    seed(get_native_id())
    path = args["dataset_bin_path"]

    with open(args["processed_repos_path"], "a+") as processed_repos_file, \
            open(args["funcs_map_path"], "a+") as map_file:
        processed_repos_file.seek(0)
        processed_repos = set(processed_repos_file.read().split("\n"))
        repositories = g.search_repositories(query=f"stars:<={stars.value} language:rust", sort="stars", order="desc")
        prev_idx = -1
        while num_files.value < args["num_files"]:
            prev_idx, repo, repositories = safe_call(p_lock, t_lock, get_repo, idx, stars, prev_idx, g, repositories)
            full_name = repo.full_name
            if full_name in processed_repos:
                continue
            ct = max_retries + 1
            while ct := ct - 1:
                try:
                    r = requests.get(repo.html_url + "/zipball/master")
                    break
                except:
                    sleep(time_to_wait)
            else:
                continue
            if not r.ok:
                continue

            bpath = os.path.join(path, repo.owner.login)
            os.makedirs(bpath, exist_ok=True)
            path_zip = os.path.join(bpath, repo.name + "-master.zip")
            with open(path_zip, "wb") as f:
                f.write(r.content)
            with ZipFile(path_zip, "r") as zip_ref:
                rname = zip_ref.filelist[0].filename
                zip_ref.extractall(bpath)
            os.remove(path_zip)
            path_proj = os.path.join(bpath, rname)
            nfiles, source_paths, bin2funcs = process_project(path_proj, args)
            with p_lock, t_lock:
                num_files.value += nfiles
                print(f"{full_name}: {nfiles}")
            if not nfiles:
                rmtree(path_proj)
                try:
                    os.rmdir(bpath)
                except OSError:
                    pass
            else:
                remove_reduntant_files(path_proj, set(chain(source_paths, bin2funcs)))
                safe_call(p_lock, t_lock, map_file.write, dumps(bin2funcs, indent=4))
                safe_call(p_lock, t_lock, map_file.write, "\n")
                map_file.flush()
            safe_call(p_lock, t_lock, processed_repos_file.write, full_name + "\n")
            processed_repos_file.flush()


def spawn_threads(p_lock, num_files, idx, stars, args):
    t_lock = tLock()
    pool = [Thread(target=gen_dataset, args=(p_lock, t_lock, num_files, idx, stars, args))
            for _ in range(args["num_threads"])]
    for t in pool:
        t.start()
    for t in pool:
        t.join()


def spawn_processes(args):
    p_lock = mLock()
    num_files = Value("i", 0, lock=False)
    idx = Value("i", -1, lock=False)
    try:
        with open(args["processed_repos_path"], "r") as processed_repos_file:
            last_repo_name = processed_repos_file.read().split("\n")[-2]
        stars = Github(args["token"]).search_repositories(query=f"repo:{last_repo_name}")[0].stargazers_count

    except FileNotFoundError:
        stars = int(1e6)

    stars = Value("i", stars, lock=False)
    pool = [Process(target=spawn_threads, args=(p_lock, num_files, idx, stars, args))
            for _ in range(args["num_processes"])]
    for p in pool:
        p.start()
    for p in pool:
        p.join()


def create_dataset_tree(path):
    for l in classes:
        os.makedirs(os.path.join(path, l), exist_ok=True)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--token", type=str, required=True)
    parser.add_argument("--dataset-bin-path", type=str, required=True)
    parser.add_argument("--dataset-ll-path", type=str, required=True)
    parser.add_argument("--extractor-path", type=str, help="path to rust_extractor binary", required=True)
    parser.add_argument("--processed-repos-path", type=str, required=True)
    parser.add_argument("--funcs-map-path", type=str, required=True)
    parser.add_argument("--ida-path", type=str, required=True)
    parser.add_argument("--get-cfg-path", type=str, help="path to get_cfg.py script from mcsema", required=True)
    parser.add_argument("--mcsema-lift-path", type=str, default="")
    parser.add_argument("--llvm-version", type=int, default=12)
    parser.add_argument("--os-version", type=str, default="ubuntu20.04")
    parser.add_argument("--llvm-dis-path", type=str, required=True)
    parser.add_argument("--prob-path", type=str, required=True)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--mcsema-disas-timeout", type=int, default=600)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--num_files", type=int, default=int(1e3))
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args["dataset_bin_path"], exist_ok=True)
    create_dataset_tree(args["dataset_ll_path"])
    spawn_processes(args)
