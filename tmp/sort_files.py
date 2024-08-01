import glob
import os.path

import dataset

path_sort = r'F:\free_eeg\raw'
file_list = glob.glob(os.path.join(path_sort, '**', '**.fif'), recursive=True)
for file in file_list:
    if not file.endswith('merge.edf'):
        print(file)
        os.remove(file)
