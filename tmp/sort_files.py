import glob
import os.path

import dataset

file_list = glob.glob(os.path.join(dataset.data_path['raw'], '**', '**.edf'), recursive=True)
for file in file_list:
    if not file.endswith('merge.edf'):
        print(file)
        os.remove(file)
