
def removeLevel(d, level):
    if type(d) != type({}):
        return d

    if level == 0:
        removed = {}
        for k, v in d.items():
            if type(v) != type({}):
                continue
            for kk, vv in v.items():
                removed[kk] = vv
        return removed

    removed = {}
    for k, v in d.items():
        removed[k] = removeLevel(v, level-1)
    return removed

def merge(a, b, path=None):
    "merges b into a"
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass # same leaf value
            else:
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a