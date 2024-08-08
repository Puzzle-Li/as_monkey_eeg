import datetime
import glob
import os

path_video = r'D:\Data\as_monkeys\free_videos'
files = glob.glob(os.path.join(path_video, '*.mkv'), recursive=True)

for file in files:
    fname = os.path.split(file)[1]
    datestr = os.path.splitext(fname)[0]
    date = datetime.datetime.strptime(datestr, "%Y-%m-%d %H-%M-%S")
    datestr2 = datetime.datetime.strftime(date, "%d-%b-%Y").upper()
    datestr3 = datetime.datetime.strftime(date, "%Hh%Mm%S.000s")
    fname_new = f'{datestr}_{datestr2}_{datestr3}.mkv'
    os.rename(file, os.path.join(os.path.split(file)[0], fname_new))

