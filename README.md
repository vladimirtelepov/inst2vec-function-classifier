## Create dataset of lifted binary functions for classification task from real word projects.

### Limitations  
- works only on Linux  
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
- mcsema  
- wine  
- ida-pro7.6.exe

### Configure wine to work with mcsema and IDA under Linux
- ```$> winecfg win10```  
- install python3.9 via wine cmd  
- install protobuf, google, yara via pip in wine cmd
- ```$> cp venv/lib/python3.8/site-packages/mcsema_disass/ida7/*.py ida_root/python/3```
- copy venv/lib/python3.8/site-packages/mcsema_disass/ida7/get_cfg.py to convenient place