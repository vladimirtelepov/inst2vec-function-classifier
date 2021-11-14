import functools
import operator
import os
import re
import mmap
import time
import magic
import queue
import github
import random
import zipfile
import spacy
import requests
import argparse
import tempfile
import threading
import subprocess
import dill as pickle
import numpy as np
import itertools as it
import multiprocessing as mp
from shutil import move, rmtree

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from extract_funcs import parse_json


def extract_funcs_from_ll(path_ll, name2label, path_dataset, randlen=5):
    one_from_names = rf"(?:{'|'.join(name2label.keys())})"
    str_start_pattern = rf"define internal %struct\.Memory\* @sub_\w+({one_from_names})"
    str_pattern = str_start_pattern + r"[a-zA-Z0-9]{20}[^}]+}\n"
    pattern = re.compile(bytes(str_pattern, encoding="utf-8"), re.ASCII | re.DOTALL)
    num_files = 0

    with open(path_ll) as llfile:
        with mmap.mmap(llfile.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_obj:
            for match in pattern.finditer(mmap_obj):
                name = str(match.group(1), encoding="utf8")
                rand = "".join(random.choice([str(i) for i in range(10)]) for _ in range(randlen))
                with open(os.path.join(os.path.join(path_dataset, name2label[name]), name + rand), "wb") as f:
                    f.write(match.group(0))
                num_files += 1

    return num_files


def predict_label(signatures, prob3_path, vocabulary_path, idf_path, cluster_centers_path, clasnum2labels_path):
    func_names = []
    func_types = []
    func_comments = []
    for name, types, comments in (x for x in map(lambda l: l.strip().split("|"), signatures)):
        func_types.append(types.split(","))
        func_comments.append(comments.split(","))
        func_names.append(name)

    def tokenize_funcs(funcs):
        oneword = re.compile(r"^[a-z][a-z0-9]+|[A-Z][A-Z0-9]$")
        difCase = re.compile(r".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)")
        under_scores_split = re.compile(r"_")

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
    tokenized_func_names = [tok_name + tok_comm for tok_name, tok_comm in zip(tokenized_func_names, func_comments)]

    def drop_wrong_symbols(tokenized_func_names):
        # first approach to drop all digits, second only if > 50%
        wrong_char = re.compile(r"[\d]")
        tokenized_func_names_ = []
        for tokenized_name in tokenized_func_names:
            processed_tokens = [wrong_char.sub("", token).lower() for token in tokenized_name if
                                wrong_char.sub("", token)]
            tokenized_func_names_.append(processed_tokens)

        return tokenized_func_names_

    tokenized_func_names = drop_wrong_symbols(tokenized_func_names)

    with open(prob3_path, "rb") as f:
        prob3 = pickle.load(f)

    def split(word, start=1, end=20):
        return ((word[:i], word[i:]) for i in range(start, min(len(word) + 1, end)))

    @functools.lru_cache(maxsize=10000)
    def segment(word, maxlen=500):
        if not word:
            return []
        if len(word) > maxlen:
            return segment(word[:maxlen]) + segment(word[maxlen:])
        candidates = ([first] + segment(remaining) for first, remaining in split(word))
        return max(candidates, key=lambda x: functools.reduce(operator.__mul__, map(prob3, x), 1))

    def segmentize_corpus(tokenized_func_names, segmenter):
        tokenized_func_names = [list(it.chain(*(segmenter(token) for token in tokens)))
                                for tokens in tokenized_func_names]
        return tokenized_func_names

    tokenized_func_names = segmentize_corpus(tokenized_func_names, segment)

    def lemmatize_corpus(tokenized_func_names):
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        lengths = np.cumsum([0] + list(map(len, tokenized_func_names)))
        flat_tokens = list(it.chain(*tokenized_func_names))
        doc = spacy.tokens.Doc(nlp.vocab, words=flat_tokens)
        tokenized_func_names = [[token.lemma_ for token in doc[lengths[i - 1]: lengths[i]]]
                                for i in range(1, len(tokenized_func_names) + 1)]
        return tokenized_func_names

    tokenized_func_names = lemmatize_corpus(tokenized_func_names)

    with open(vocabulary_path, "rb") as f:
        vocab = pickle.load(f)

    def prune_names(tokenized_func_names, vocab):
        tokenized_func_names_ = []
        for tokenized_name in tokenized_func_names:
            processed_tokens = [token for token in tokenized_name if token in vocab]
            tokenized_func_names_.append(processed_tokens)
        return tokenized_func_names_

    tokenized_func_names = prune_names(tokenized_func_names, set(vocab.keys()))

    def tokenize_types(func_types):
        type_set = {
            "int", "unsigned int", "char", "unsigned char", "enum", "struct", "void", "long", "unsigned long",
            "float", "double", "short", "unsigned short", "bool", "union", "long long", "unsigned long long"}
        type_dict = {re.compile(t): t for t in type_set}
        re_drop = re.compile(r"\*|restrict|const")
        struct_type = re.compile("struct")
        tokenized_types = [[0 for _ in range(len(f_types))] for f_types in func_types]
        for i, f_types in enumerate(func_types):
            for j, type in enumerate(f_types):
                cleaned_type = re_drop.sub("", type)
                for re_t, t in type_dict.items():
                    if re.search(re_t, cleaned_type):
                        tokenized_types[i][j] = t
                        break
                else:
                    tokenized_types[i][j] = type_dict[struct_type]
        return tokenized_types

    tokenized_func_types = tokenize_types(func_types)

    tokenized_features = [tok_name + tok_types for tok_name, tok_types in
                          zip(tokenized_func_names, tokenized_func_types)]
    idf = np.load(idf_path)
    tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False, sublinear_tf=True, vocabulary=vocab)
    tfidf_vectorizer.idf_ = idf
    tfidf_matrix = tfidf_vectorizer.transform(tokenized_features)

    centers = np.load(cluster_centers_path)
    model = KMeans(centers.shape[0])
    model._n_threads = 1
    model.cluster_centers_ = centers
    cluster_nums = model.predict(tfidf_matrix)

    with open(clasnum2labels_path, "rb") as f:
        clasnum2labels = pickle.load(f)

    labels = [clasnum2labels[c] for c in cluster_nums]
    return {n: l for n, l in zip(func_names, labels) if l != "unknown"}


def lift(path_bin, path_ida, path_mcsema_lift, path_llvm_dis, mcsema_disas_timeout, max_bin_size=int(500e6)):
    if os.stat(path_bin).st_size > max_bin_size:
        return
    path_cfg, path_bc, path_ll = f"{path_bin}.cfg", f"{path_bin}.bc", f"{path_bin}.ll"
    try:
        ret_code = subprocess.call(f"wine {path_ida} -B -S\"{args['get_cfg_path']} --output {path_cfg} --arch amd64 "
                                   f"--os linux --entrypoint main\" {path_bin}", shell=True,
                                   timeout=mcsema_disas_timeout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if any((ret_code, not os.path.exists(path_cfg), not os.stat(path_cfg).st_size)):
            return
    except subprocess.TimeoutExpired:
        return
    if subprocess.call(f"{path_mcsema_lift} --output {path_bc} --arch amd64 --os linux --cfg {path_cfg}", shell=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL):
        return
    if subprocess.call(f"{path_llvm_dis} {path_bc} -o {path_ll}", shell=True):
        return
    return path_ll if os.path.exists(path_ll) and os.stat(path_ll).st_size else None


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
            for p in os.scandir(cur):
                q.put(p)

    signatures = set()
    with tempfile.NamedTemporaryFile("r+") as tmpfile:
        for p in source_paths:
            tmpfile.truncate(0)
            if subprocess.call(f"{args['extractor_path']} {p} {tmpfile.name}", shell=True, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL):
                continue
            tmpfile.seek(0)
            signatures |= set(parse_json(tmpfile))
    name2label = predict_label(signatures, args["prob3_path"], args["vocabulary_path"], args["idf_path"],
                               args["cluster_centers_path"], args["clasnum2labels_path"])

    os.chdir(path)
    num_files = 0
    if subprocess.call("cargo build --all-features", shell=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL):
        return num_files

    binary_paths = []
    for f in os.scandir(os.path.join(os.path.join(path, "target"), "debug")):
        if f.is_file():
            file_info = magic.from_file(os.path.abspath(f))
            if file_info == "current ar archive" or file_info.startswith("ELF"):
                binary_paths.append(os.path.abspath(f))

    for p in binary_paths:
        with tempfile.TemporaryDirectory() as tempdir:
            move(p, tempdir)
            path_ll = lift(os.path.join(tempdir, os.path.basename(p)), args["ida_path"], args["mcsema_lift_path"],
                           args["llvm_dis_path"], args["mcsema_disas_timeout"])
            if path_ll:
                num_files += extract_funcs_from_ll(path_ll, name2label, args["dataset_path"])
    return num_files


def safe_call(p_lock, t_lock, g, f, *args, **kwargs):
    with p_lock, t_lock:
        remaining, _ = g.rate_limiting
        if not remaining:
            time.sleep(abs(g.rate_limiting_resettime - time.time()))
        res = f(*args, **kwargs)
    return res


def gen_dataset(p_lock, t_lock, p_num, t_num, g, args, max_retries=3, time_to_wait=5):
    random.seed(threading.get_native_id())
    num_files = 0
    stars = int(1e6)
    total_num_threads = args["num_threads"] * args["num_processes"]
    thread_idx = p_num * args["num_threads"] + t_num

    with open(args["processed_repos_path"], "a+") as processed_repos_file, tempfile.TemporaryDirectory() as tmpdir:
        processed_repos_file.seek(0)
        processed_repos = set(processed_repos_file.read().split("\n"))
        while num_files < args["num_files"]:
            repositories = g.search_repositories(query=f"stars:<={stars} language:rust", sort="stars", order="desc")
            num_repos = repositories.totalCount
            count_per_thread = num_repos // total_num_threads
            rg = range(count_per_thread * thread_idx, count_per_thread * (thread_idx + 1)) \
                if thread_idx != total_num_threads - 1 else range(count_per_thread * thread_idx, num_repos)

            for ind in rg:
                repo = safe_call(p_lock, t_lock, g, repositories.__getitem__, ind)
                stars = repo.stargazers_count
                if num_files > args["num_files"]:
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

                path_zip = os.path.join(tmpdir, full_name + ".zip")
                with open(path_zip, "wb") as f:
                    f.write(r.content)
                with zipfile.ZipFile(path_zip, "r") as zip_ref:
                    zip_ref.extractall(tmpdir)
                    num_files += process_project(os.path.join(tmpdir, repo.name + "-master"), args)
                rmtree(os.path.join(tmpdir, repo.name + "-master"))
                os.remove(path_zip)
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


def create_dataset_tree(path, clasnum2labels_path):
    with open(clasnum2labels_path, "rb") as f:
        clasnum2labels = pickle.load(f)
    labels = clasnum2labels.values()
    for l in labels:
        if l != "unknown":
            os.makedirs(os.path.join(path, l), exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--extractor-path", type=str, help="path to rust_extractor binary", required=True)
    parser.add_argument("--processed-repos-path", type=str, required=True)
    parser.add_argument("--ida-path", type=str, required=True)
    parser.add_argument("--get-cfg-path", type=str, help="path to get_cfg.py script from mcsema", required=True)
    parser.add_argument("--mcsema-lift-path", type=str, required=True)
    parser.add_argument("--llvm-dis-path", type=str, required=True)
    parser.add_argument("--prob3-path", type=str, required=True)
    parser.add_argument("--vocabulary-path", type=str, required=True)
    parser.add_argument("--idf-path", type=str, required=True)
    parser.add_argument("--cluster-centers-path", type=str, required=True)
    parser.add_argument("--clasnum2labels-path", type=str, required=True)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--mcsema-disas-timeout", type=int, default=600)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--num_files", type=int, default=int(1e3), help="max generated .ll files per thread")
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    args = parse_args()
    create_dataset_tree(args["dataset_path"], args["clasnum2labels_path"])
    spawn_processes(args)
