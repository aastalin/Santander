"""
Data parser for Santander competitation at Kaggle
Written by Aasta Lin
BSD License
"""
import csv
import os

# data I/O
rptr = open('train_ver2.csv',"r")
reader = csv.reader(rptr)
header = next(reader) #header

while True:
    try:
        line = next(reader)
    except StopIteration:
        break
    idx = int(line[1])
    name = "%.3d/user_%.7d.csv" % (idx/10000, idx)

    if os.path.isfile(name):
        wptr = open(name, "a")
        writer = csv.writer(wptr)
        writer.writerow(line)
        wptr.close()
    else:
        wptr = open(name, "w")
        writer = csv.writer(wptr)
        writer.writerow(header)
        writer.writerow(line)
        wptr.close()
