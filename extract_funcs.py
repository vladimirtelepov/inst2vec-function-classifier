import os
import argparse
import clang.cindex
import multiprocessing as mp
from threading import Thread
import itertools


def _extract_funcs(q_in, q_out, args):
    funcs = set()
    index = clang.cindex.Index.create()
    while not q_in.empty():
        folder_path = os.path.join(args["path_in"], q_in.get())

        for file in os.scandir(folder_path):
            try:
                translation_units = index.parse(file.path, args=["-O0"])
            except:
                continue

            func_nodes = (node for node in translation_units.cursor.get_children() if
                          node.kind == clang.cindex.CursorKind.FUNCTION_DECL)
            funcs |= set(",".join(
                itertools.chain((f.spelling, f.result_type.get_canonical().spelling),
                                (arg.type.get_canonical().spelling for arg in f.get_arguments()))
            ) for f in func_nodes)

    q_out.put(funcs)


def extract_funcs(q_in, q_out, args):
    pool = [Thread(target=_extract_funcs, args=(q_in, q_out, args)) for _ in range(args["num_threads"])]
    for t in pool:
        t.start()
    for t in pool:
        t.join()


def main(args):
    q_in = mp.Queue()
    q_out = mp.Queue()
    for f in os.listdir(args["path_in"]):
        q_in.put(f)

    num_processes = args["num_processes"]
    num_threads = args["num_threads"]

    pool = [mp.Process(target=extract_funcs, args=(q_in, q_out, args)) for _ in range(num_processes)]
    for p in pool:
        p.start()

    funcs = set()
    for i in range(num_processes * num_threads):
        funcs |= q_out.get()

    for p in pool:
        p.join()

    with open(f"{args['file']}", "w") as out:
        out.write("\n".join(funcs))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-in", type=str, required=True)
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--num-threads", type=int, required=True)
    parser.add_argument("--num-processes", type=int, required=True)
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
