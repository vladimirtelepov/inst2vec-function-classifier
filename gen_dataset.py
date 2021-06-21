import json
import github
import requests
import os
import zipfile
import argparse
import time
import tempfile
import multiprocessing as mp
import threading
import subprocess
import magic
import queue
from extract_funcs import parse_json


def createdir(dname):
    try:
        os.makedirs(dname)
    except FileExistsError:
        pass


def get_dir_size(path):
    size = 0
    for p in os.scandir(path):
        size += p.stat().st_size if p.is_file() else get_dir_size(p.path)
    return size


def extract_funcs(ll_path, sign2labels, args):
    pass


def predict_label(signatures):
    pass


def process_project(path, args):
    source_paths = []
    q = queue.Queue()
    for p in os.scandir(path):
        q.put(p)
    while not q.empty():
        cur = q.get()
        if cur.is_file() and cur.name.endswith(".rs"):
            source_paths.append(os.path.abspath(cur))
        elif cur.is_dir():
            q.put(os.scandir(cur))

    signatures = {}
    with tempfile.NamedTemporaryFile("r+") as tmpfile:
        for p in source_paths:
            tmpfile.truncate(0)
            ret_code = subprocess.run([args["extractor_path"], p, tmpfile.name])
            if not ret_code:
                return
            tmpfile.seek(0)
            signatures |= set(parse_json(tmpfile))
    sign2label = predict_label(signatures)

    os.chdir(path)
    ret_code = subprocess.run(["cargo", "build", "--all-features", "--bins", "--lib"])
    if not ret_code:
        return

    binary_paths = []
    for f in os.scandir(os.path.join(os.path.join(path, "target"), "debug")):
        if f.is_file():
            file_info = magic.from_file(f)
            if file_info == "current ar archive" or file_info.startswith("ELF"):
                binary_paths.append(os.path.abspath(f))

    for p in binary_paths:
        cfg_path = os.path.join(path, p.name + ".cfg")
        bc_path = os.path.join(path, p.name + ".bc")
        ll_path = os.path.join(path, p.name + ".ll")
        ret_code = subprocess.run(["wine", f"\"{args['ida_path']}\"", "-B",
                                   f"-S\"\\\"{args['get_cfg_path']}\\\" --output {cfg_path}"
                                   "--arch amd64_avx --os linux --entrypoint main\"", p])
        if not ret_code:
            return
        ret_code = subprocess.run([f"{args['mcsema_lift_path']}", "--output", f"{bc_path}", "--arch", "amd64_avx",
                                   "--os", "linux", "--cfg", f"{cfg_path}"])
        if not ret_code:
            return
        ret_code = subprocess.run([f"{args['llvm_dis_path']}", f"{bc_path}", "-o", f"{ll_path}"])
        if not ret_code:
            return
        extract_funcs(ll_path, sign2label, args)


def safe_call(p_lock, t_lock, g, f, *args, **kwargs):
    p_lock.acquire()
    t_lock.acquire()
    remaining, _ = g.rate_limiting
    if not remaining:
        time.sleep(abs(g.rate_limiting_resettime - time.time()))
    res = f(*args, **kwargs)
    t_lock.release()
    p_lock.release()
    return res


def gen_dataset(p_lock, t_lock, p_num, t_num, g, args):
    max_size = args["maxsize"] * 2 ** 30
    max_retries = 3
    time_to_wait = 5
    size = 0
    stars = int(1e6)
    total_num_threads = args["num_threads"] * args["num_processes"]
    thread_idx = p_num * args["num_threads"] + t_num
    processed_repos_file = open(args["processed_repos_path"])
    processed_repos = set(processed_repos_file.read().split("\n"))
    tmpdir = tempfile.TemporaryDirectory()
    path = tmpdir.name

    while True:
        repositories = g.search_repositories(query=f"stars:<={stars} language:rust", sort="stars", order="desc")
        num_repos = repositories.total_count
        count_per_thread = num_repos // total_num_threads
        rg = range(count_per_thread * thread_idx, count_per_thread * (thread_idx + 1)) \
            if thread_idx != total_num_threads - 1 else range(count_per_thread * thread_idx, num_repos)

        for ind in rg:
            repo = safe_call(p_lock, t_lock, g, repositories.__getitem__, ind)
            stars = repo.stargazers_count
            if size > max_size:
                break

            full_name = repo.full_name.replace("/", "_")
            if full_name in processed_repos:
                continue
            ct = max_retries + 1
            while ct := ct - 1:
                try:
                    r = requests.get(repo.html_url + "/archive/master.zip")
                    break
                except:
                    time.sleep(time_to_wait)
            else:
                continue
            if not r.ok:
                continue

            path_zip = os.path.join(path, full_name + ".zip")
            with open(path_zip, "wb") as f:
                f.write(r.content)
            with zipfile.ZipFile(path_zip, "r") as zip_ref:
                with tempfile.TemporaryDirectory() as tempdir:
                    zip_ref.extractall(tempdir)
                    process_project(os.path.join(tempdir.name, repo.name + "-master"), args)
            os.remove(path_zip)

            size += update_size()

            processed_repos.add(full_name)
            safe_call(p_lock, t_lock, g, processed_repos_file.write, full_name + "\n")


def spawn_threads(p_lock, p_num, g, args):
    t_lock = threading.Lock()
    pool = [threading.Thread(target=gen_dataset, args=(p_lock, t_lock, p_num, t_num, g, args))
            for t_num in range(args["num_threads"])]
    for t in pool:
        t.start()
    for t in pool:
        t.join()


def spawn_processes(args):
    g = github.Github(args["token"])
    p_lock = mp.Lock()
    pool = [mp.Process(target=spawn_threads, args=(p_lock, p_num, g, args)) for p_num in range(args["num_processes"])]
    for p in pool:
        p.start()
    for p in pool:
        p.join()


def create_dataset_tree(path, params_path):
    with open(params_path) as f:
        params = json.load(f)
    classes = [p["name"] for p in params]
    for c in classes:
        createdir(os.path.join(path, c))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--clustering-params-path", type=str, required=True)
    parser.add_argument("--extractor-path", type=str, help="path to rust_extractor binary", required=True)
    parser.add_argument("--processed-repos-path", type=str, required=True)
    parser.add_argument("--ida-path", type=str, required=True)
    parser.add_argument("--get-cfg-path", type=str, help="path to get_cfg.py script from mcsema", required=True)
    parser.add_argument("--mcsema-lift-path", type=str, required=True)
    parser.add_argument("--llvm-dis-path", type=str, required=True)
    parser.add_argument("--num-threads", type=int, required=True)
    parser.add_argument("--num-processes", type=int, required=True)
    parser.add_argument("--maxsize", type=int, help="GB", required=True)
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    args = parse_args()
    create_dataset_tree(args["dataset_path"], args["clustering_params_path"])
    spawn_processes(args)
