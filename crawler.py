import github
import requests
import os
import zipfile
import argparse
import shutil
import time


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


def extract_files(path_in, path_out, lang):
    for p in os.scandir(path_in):
        if p.is_dir():
            extract_files(p.path, path_out, lang)
        elif lang == "c++" and any((p.name.endswith(".cpp"), p.name.endswith(".cc"), p.name.endswith(".c"))) or \
                lang == "rust" and p.name.endswith(".rs"):
            shutil.copy2(p, path_out)


def crawl(args):
    token = args["token"]
    path = args["path"]
    store = args["store"]
    max_size = args["maxsize"] * 2 ** 30
    max_retries = 3
    time_to_wait = 5
    g = github.Github(token)
    size = 0
    stars = int(1e6)
    createdir(path)
    # TODO: use threads
    while size < max_size:
        remaining, _ = g.rate_limiting
        if not remaining:
            time.sleep(g.rate_limiting_resettime - time.time())
        repositories = g.search_repositories(
            query=f"stars:<={stars} {'language:c language:c++' if args['language'] == 'c++' else 'language:rust'}",
            sort="stars",
            order="desc")
        for repo in repositories:
            stars = repo.stargazers_count
            if size > max_size:
                break

            remaining, _ = g.rate_limiting
            if not remaining:
                time.sleep(g.rate_limiting_resettime - time.time())

            path_funcs = os.path.join(path, repo.name)
            path_master = os.path.join(path, repo.name + "-master")
            if os.path.exists(path_funcs):
                size += get_dir_size(path_funcs)
                if store:
                    size += get_dir_size(path_master)
                continue

            ct = max_retries + 1
            while ct := ct - 1:
                try:
                    r = requests.get(repo.html_url + "/archive/master.zip")
                    break
                except:
                    time.sleep(time_to_wait)
            else:
                print(repo.html_url)
                continue

            if not r.ok:
                continue

            path_zip = os.path.join(path, repo.name + ".zip")
            with open(path_zip, "wb") as f:
                f.write(r.content)
            with zipfile.ZipFile(path_zip, "r") as zip_ref:
                zip_ref.extractall(path)
                dir_size = sum(el.file_size for el in zip_ref.infolist())
            os.remove(path_zip)

            createdir(path_funcs)
            extract_files(path_master, path_funcs, args["language"])
            size += get_dir_size(path_funcs)
            if store:
                size += dir_size
            else:
                shutil.rmtree(path_master)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, required=True)
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--store", help="store crawled repo", action="store_true")
    parser.add_argument("--maxsize", type=int, help="GB", required=True)
    parser.add_argument("--language", type=str, help="c++ or rust", required=True)
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    args = parse_args()
    crawl(args)
