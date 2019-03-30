# -*- coding:utf-8 -*-

import os

root_dir = os.path.dirname(__file__)
data_dir = os.path.join(root_dir, 'data')
raw_dir = os.path.join(root_dir, 'raw')
save_dir = os.path.join(root_dir, 'save')

sxhy_path = os.path.join(data_dir, 'sxhy_dict.txt')
char_dict_path = os.path.join(data_dir, 'char_dict.txt')
poems_path = os.path.join(data_dir, 'poem.txt')
wordrank_path = os.path.join(data_dir, 'wordrank.txt')
plan_data_path = os.path.join(data_dir, 'plan_data.txt')
plan_history_path = os.path.join(data_dir, 'plan_history.txt')

_dependency_dict = {
    poems_path: [char_dict_path],
    wordrank_path: [sxhy_path, poems_path],
    plan_history_path: [char_dict_path, poems_path, sxhy_path],
    plan_data_path: [char_dict_path, poems_path, sxhy_path],
}


def check_uptodate(path):
    """ Return true iff the file exists and up-to-date with dependencies."""
    if not os.path.exists(path):
        # File not found.
        return False
    timestamp = os.path.getmtime(path)
    if path in _dependency_dict:
        for dependency in _dependency_dict[path]:
            if not os.path.exists(dependency) or \
                    os.path.getmtime(dependency) > timestamp:
                # File stale.
                return False
    return True
