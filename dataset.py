import os.path

import pandas as pd

path = {'root': r'D:\Data\as_monkeys'}

path['raw'] = os.path.join(path['root'], 'raw', 'X8_data')
path['tmp'] = os.path.join(path['root'], 'tmp')
path['results'] = os.path.join(path['root'], 'results')

for folder in path.values():
    if folder == 'root':
        continue

    if not os.path.exists(folder):
        os.makedirs(folder)


def load_info():
    return pd.read_excel(os.path.join(path['root'], 'monkeys.xlsx'), index_col=0)
