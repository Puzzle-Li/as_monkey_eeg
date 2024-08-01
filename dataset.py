import os.path

import pandas as pd

path = {'root': r'D:\Data\as_monkeys'}
path['free_eeg'] = os.path.join(path['root'], 'free_eeg')
path['raw'] = os.path.join(path['free_eeg'], 'raw')
path['tmp'] = os.path.join(path['free_eeg'], 'tmp')
path['tbl'] = os.path.join(path['free_eeg'], 'tbl')
path['fig'] = os.path.join(path['free_eeg'], 'fig')

for folder in path.values():
    if folder == 'root':
        continue

    if not os.path.exists(folder):
        os.makedirs(folder)


def load_info():
    return pd.read_excel(os.path.join(path['root'], 'monkeys.xlsx'), index_col=0)
