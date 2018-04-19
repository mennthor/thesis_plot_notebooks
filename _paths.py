# coding: utf-8

"""
Working directories for each branch.
By using the paths stored here consistently, we can set new working directories
for each branch, without interfering with previous results.

In a script use `from PATHS import PATHS` and eg. `local_path = PATHS.local`.
Get all available paths with `print(PATHS)`.
```
"""

import os as _os
from git import Repo as _Repo


class _Paths(object):
    """
    Class to acces paths via it's attributes.
    Code adopted from scipy.optimize.OptimizeResult.
    """
    def __init__(self, d):
        self._d = d

    def __getattr__(self, name):
            try:
                return self._d[name]
            except KeyError:
                raise AttributeError(name)

    def __setattr__(self, name, val):
        if name != "_d":
            raise RuntimeError("PATHS is readonly.")
        else:
            # We could still overwrite _d, but who does that?
            super(_Paths, self).__setattr__(name, val)

    def __repr__(self):
        m = max(map(len, list(self._d.keys()))) + 1
        return '\n'.join([name.rjust(m) + ': ' + path
                          for name, path in self._d.items()])


# Insert the current branch name to automatically switch to a new work dir
_repo_path = _os.path.abspath("./")
_repo_name = _os.path.basename(_repo_path)
_repo = _Repo(_repo_path)
_BRANCH_NAME = _repo.active_branch.name

_data_path = _os.path.join("/Users", "tmenne", "Downloads",
                           "hese_transient_stacking_data")

_paths = {
    "repo": _repo_path,
    "local": _os.path.join(_data_path, "out_master"),
    "data": _os.path.join(_data_path, "rawout_master"),
    "skylab_data": _os.path.join(_data_path, "skylab_data"),
    "plots": _os.path.join(_repo_path, _BRANCH_NAME + "plots")
}

PATHS = _Paths(_paths)

# Make another object for the paths with the original HESE source files
_paths_orig = {
    "repo": _repo_path,
    "local": _os.path.join(_data_path, "out_original_hese"),
    "data": _os.path.join(_data_path, "rawout_original_hese"),
    "skylab_data": _os.path.join(_data_path, "skylab_data"),
    "plots": _os.path.join(_repo_path, _BRANCH_NAME + "plots")
}
PATHS_ORIG = _Paths(_paths_orig)
