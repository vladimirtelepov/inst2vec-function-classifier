## Create dataset of lifted binary functions for classification task from real word projects.

### Limitations  
- works only on amd64 Linux  
- extract_funcs.py can exceed memory limits because libclang library  
suggested do not run it on files from next repositories:  
llvm-project, gcc, src, clang  

### Requirements
- libclang-11-dev  
- python3.8  
- python packages from requirements.txt (suggested to use venv)
- spacy en model  
- ntlk stopwords  
- github personal access token
- mcsema (native or docker version)  
- wine  
- ida-pro7.6 (for Windows)

### Configure wine to work with mcsema-lift and IDA(Windows version) under Linux
- ```$> winecfg win10```  
- install python3.9 via wine cmd  
- install protobuf, google, yara via pip in wine cmd
- download scripts from https://github.com/lifting-bits/mcsema/tree/master/tools/mcsema_disass/ida7 and move to 
  ida_root/python/3```
- copy get_cfg.py to convenient place