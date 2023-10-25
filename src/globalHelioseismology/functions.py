import os

def mkdirs(fullpath):
    if not os.path.exists(fullpath): os.makedirs(fullpath)
