import os
import argparse
import clang.cindex as cind
import multiprocessing as mp
import threading
import subprocess
import tempfile
import itertools
import re
import queue
import json


re_parse = re.compile(r"[a-zA-Z0-9]+")


def extract_funcs_cpp(q_in, q_out, args):
    func_types = [cind.CursorKind.FUNCTION_DECL, cind.CursorKind.CXX_METHOD, cind.CursorKind.FUNCTION_TEMPLATE]
    struct_types = [cind.CursorKind.CLASS_DECL, cind.CursorKind.STRUCT_DECL, cind.CursorKind.CLASS_TEMPLATE]
    funcs = set()
    index = cind.Index.create()
    while True:
        try:
            cur_folder = q_in.get(timeout=1)
        except queue.Empty:
            break

        folder_path = os.path.join(args["path_in"], cur_folder)
        for file in os.scandir(folder_path):
            try:
                translation_units = index.parse(file.path, args=["-O0", "-fparse-all-comments"])
            except:
                continue

            q = queue.Queue()
            for node in translation_units.cursor.get_children():
                if node.kind in struct_types or node.kind in func_types:
                    q.put(node)

            func_nodes = []
            while not q.empty():
                cur = q.get()
                if cur.kind in func_types:
                    func_nodes.append(cur)
                elif cur.kind in struct_types:
                    for node in cur.get_children():
                        q.put(node)

            funcs_from_file = []
            for f in func_nodes:
                try:
                    comment = f.raw_comment
                except UnicodeDecodeError:
                    comment = ""

                arg_types = (arg.type.get_canonical().spelling for arg in f.get_children()
                             if arg.kind == cind.CursorKind.PARM_DECL) \
                    if f.kind == cind.CursorKind.FUNCTION_TEMPLATE \
                    else (arg.type.get_canonical().spelling for arg in f.get_arguments())
                types_string = ",".join(itertools.chain([f.result_type.get_canonical().spelling], arg_types))

                funcs_from_file.append("|".join(
                    [f.spelling, types_string,
                     ",".join(map(lambda x: x.group(), re_parse.finditer(comment))) if comment else ""]))
            funcs |= set(funcs_from_file)

    q_out.put(funcs)


def change_type_cpp_analog(t):
    rust2cpp = {
        "i8": "char",
        "i16": "short",
        "i32": "int",
        "i64": "long",
        "i128": "long long",
        "u8": "unsigned char",
        "u16": "unsigned short",
        "u32": "unsigned int",
        "u64": "unsigned long",
        "u128": "unsigned long long",
        "f32": "float",
        "f64": "double",
        "usize": "unsigned long",
        "()": "void",
        "enum": "enum",
        "union": "union",
        "bool": "bool"
    }
    return rust2cpp[t] if t in rust2cpp else "struct"


def parse_json(json_file):
    funcs_from_file = []
    try:
        funcs = json.load(json_file)
    except json.decoder.JSONDecodeError:
        return set()

    for f in funcs:
        comment = " ".join(f["comments"])
        funcs_from_file.append("|".join(
            [f["name"],
             ",".join(itertools.chain([change_type_cpp_analog(f["output_type"])],
                                      map(change_type_cpp_analog, f["arg_types"]))),
             ",".join(map(lambda x: x.group(), re_parse.finditer(comment)) if comment else "")]))
    return funcs_from_file


def extract_funcs_rust(q_in, q_out, args):
    funcs = set()
    while True:
        try:
            cur_folder = q_in.get(timeout=1)
        except queue.Empty:
            break

        folder_path = os.path.join(args["path_in"], cur_folder)
        for file in os.scandir(folder_path):
            with tempfile.NamedTemporaryFile("r+") as tmp:
                subprocess.run([args["path_to_extractor"], file.path, tmp.name])
                tmp.seek(0)
                funcs_from_file = parse_json(tmp)
            funcs |= set(funcs_from_file)

    q_out.put(funcs)


def extract_funcs(q_in, q_out, args):
    pool = [threading.Thread(target=extract_funcs_cpp if args["language"] == "c++" else extract_funcs_rust,
                             args=(q_in, q_out, args)) for _ in range(args["num_threads"])]
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
    parser.add_argument("--language", type=str, help="c++ or rust", required=True)
    parser.add_argument("--path-to-extractor", type=str, help="path to rust_extractor binary")
    args = parser.parse_args()
    if args.language == "rust" and args.path_to_extractor is None:
        raise Exception("You must specify --path-to-extractor argument")
    return vars(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
