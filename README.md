## Prerequirements
libclang-11-dev  
python3.8  
spacy en model  
github personal access token
mcsema  
wine  
ida-pro7.5

## Limitations  
extract_funcs.py can exceed memory limits because libclang library  
suggested do not run it on files from next repositories:  
llvm-project, gcc, src, clang  

## Mcsema pre-installation steps
install wine  
download ida7.5  
install python3.8.5 via cmd in wine  
install protobuf, google via pip in wine