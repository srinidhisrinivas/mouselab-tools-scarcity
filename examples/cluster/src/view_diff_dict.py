import sys
import dill as pickle
from pathlib import Path
import json

try:
    file_path = sys.argv[1]
except:
    raise "Filename not present"

try:
    number = sys.argv[2]
except:
    number = 20
        
path = (
        Path(__file__)
        .resolve()
        .parents[1]
        .joinpath(file_path)
    )

try:
    with open(path, "rb") as f:
        loaded = pickle.load(f)
        for i, (k,v) in enumerate(loaded.items()):
            print("\n{}\n1: {} \n2: {}\n".format(k, v["1"], v["2"]))
            if (i+1) > number:
                break
except:
    raise "File cannot be read: {}".format(path)
