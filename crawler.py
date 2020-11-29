import github
import requests
import os
import zipfile
import argparse
import shutil


def createdir(dname):
    try:
        os.makedirs(dname)
    except FileExistsError:
        pass


def get_dir_size(path):
    size = 0
    for p in os.scandir(path):
        if p.is_file():
            size += p.stat().st_size
        elif p.is_dir():
            size += get_dir_size(p.path)
    return size


def extract_cpp_files(path_in, path_out):
    createdir(path_out)
    for p in os.scandir(path_in):
        if p.is_dir():
            extract_cpp_files(p.path, path_out)
        elif p.name.endswith(".cpp") or p.name.endswith(".cc") or p.name.endswith(".c"):
            shutil.copy2(p, path_out)


def crawl(args):
    token = args["token"]
    path = args["path"]
    store = args["store"]
    max_size = args["maxsize"] * 2**30

    g = github.Github(token)
    repositories = g.search_repositories(
        query="language:c++",
        sort="stars",
        order="desc")
    size = 0
    for repo in repositories:
        if size > max_size:
            break
        r = requests.get(repo.html_url + "/archive/master.zip")
        path_zip = os.path.join(path, repo.name + ".zip")
        with open(path_zip, "wb") as f:
            f.write(r.content)
        try:
            with zipfile.ZipFile(path_zip, "r") as zip_ref:
                zip_ref.extractall(path)
        except:
            os.remove(path_zip)
            continue

        os.remove(path_zip)
        path_master = os.path.join(path, repo.name + "-master")
        path_funcs = os.path.join(path, repo.name)

        extract_cpp_files(path_master, path_funcs)
        size += get_dir_size(path_funcs)
        if store:
            size += get_dir_size(path_master)
        else:
            shutil.rmtree(path_master)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, required=True)
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--store", action="store_true", required=True)
    parser.add_argument("--maxsize", type=int, help="GB", required=True)
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    args = parse_args()
    crawl(args)
