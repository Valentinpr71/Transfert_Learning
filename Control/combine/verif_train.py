import os
import json
import hashlib
def check_train(dim):
    dim=json.dumps(dim,sort_keys=True).encode('utf-8')
    crypted_name=hashlib.md5(dim).hexdigest()
    return os.path.exists(os.path.join("Result",crypted_name+".zip"))