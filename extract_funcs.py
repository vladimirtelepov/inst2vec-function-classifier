import os
import argparse
import clang.cindex
import typing
from threading import Thread


class SimpleTread(Thread):
    def __init__(self, folders, file):
        Thread.__init__(self)
        self.folders = folders
        self.file = file

    def run(self):
        funcs = set()
        for folder in self.folders:
            print(f"Processeing {folder.name}")
            for file in os.scandir(folder):
                funcs |= get_func_names_from_file(file.path)

        with open(self.file, "w") as out:
            for f in funcs:
                out.write(f + " ")


def get_func_names_from_file(file_path):
    def filter_node_list_by_kind(
            nodes: typing.Iterable[clang.cindex.Cursor],
            kinds: list
    ) -> typing.Iterable[clang.cindex.Cursor]:
        result = []
        for node in nodes:
            if node.kind in kinds:
                result.append(node)

        return result

    index = clang.cindex.Index.create()
    try:
        translation_units = index.parse(file_path, args=["-O0"])
    except:
        print("failed to compile {} file".format(file_path))
        return set()

    funcs = filter_node_list_by_kind(translation_units.cursor.get_children(), [clang.cindex.CursorKind.FUNCTION_DECL])
    func_names = [func.spelling for func in funcs]
    return set(func_names)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-in", type=str, required=True)
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--num-threads", type=int, required=True)
    args = parser.parse_args()
    return vars(args)


def extract_funcs(args):
    path_in = args["path_in"]
    file = args["file"]
    num_threads = args["num_threads"]
    folders = list(os.scandir(path_in))

    threads = []
    count_per_thread = len(folders) // num_threads
    for i in range(num_threads - 1):
        threads.append(SimpleTread(folders[i * count_per_thread: (i + 1) * count_per_thread], file + f"{i}.txt"))
        threads[i].start()
    threads.append(SimpleTread(folders[(num_threads - 1) * count_per_thread:], file + f"{num_threads - 1}.txt"))
    threads[num_threads - 1].start()
    for i in range(num_threads):
        threads[i].join()

    funcs = set()
    for i in range(num_threads):
        with open(file + f"{i}.txt", "r") as f:
            funcs |= set(f.read().split(" "))
        os.remove(file + f"{i}.txt")

    with open(file + ".txt", "w") as out:
        for f in funcs:
            out.write(f + "\n")


if __name__ == "__main__":
    args = parse_args()
    extract_funcs(args)
