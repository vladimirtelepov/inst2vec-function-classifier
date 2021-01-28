## Prerequirements
libclang-11-dev  
spacy en model  
python3.8  
github personal access token

## Limitations  
extract_funcs.py can exceed memory limits because libclang library  
suggested do not run it on files from next repositories:  
llvm-project, gcc, src, clang  