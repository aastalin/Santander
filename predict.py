"""
NN for Santander competitation at Kaggle
Written by Aasta Lin
BSD License
"""

import numpy as np
import pickle
import csv
import os

fsize = 24
wptr = open('response1.csv',"w")
writer = csv.writer(wptr)

with open('weight2_20.pickle') as ff:
    [Wtt,
    Wxh_sexo0, Why_sexo0, bh_sexo0, by_sexo0,
    Wxh_sexo1, Why_sexo1, bh_sexo1, by_sexo1,
    Wxh_age0, Why_age0, bh_age0, by_age0,
    Wxh_age1, Why_age1, bh_age1, by_age1,
    Wxh_age2, Why_age2, bh_age2, by_age2,
    Wxh_age3, Why_age3, bh_age3, by_age3,
    Wxh_new0, Why_new0, bh_new0, by_new0,
    Wxh_new1, Why_new1, bh_new1, by_new1,
    Wxh_rel0, Why_rel0, bh_rel0, by_rel0,
    Wxh_rel1, Why_rel1, bh_rel1, by_rel1,
    Wxh_rel2, Why_rel2, bh_rel2, by_rel2,
    Wxh_rel3, Why_rel3, bh_rel3, by_rel3,
    Wxh_seg0, Why_seg0, bh_seg0, by_seg0,
    Wxh_seg1, Why_seg1, bh_seg1, by_seg1,
    Wxh_seg2, Why_seg2, bh_seg2, by_seg2] = pickle.load(ff)


def forward(Wxh, Why, bh, by, inputs):
    hs = np.tanh(np.dot(Wxh, inputs) + bh)
    ys = np.dot(Why,hs) + by
    return hs, ys


def process(filename):
    ptr = open(filename,"r")
    reader = csv.reader(ptr)
    header = next(reader)

    while True:
        try:
            tmp = next(reader)
            line = tmp
        except StopIteration:
            break
    for i in range(len(line)):
        if line[i]=='NA':
            output = "%d" % int(line[1])
            writer.writerow(output.split(","))
            return 0

    tmp =  np.array(line[24:49], dtype='Float32')
    state = np.reshape(tmp.astype(np.float), (fsize,1))

    # forward pass
    if line[4]=='V':
        h_sexo, y_sexo = forward(Wxh_sexo0, Why_sexo0, bh_sexo0, by_sexo0, state)
    else:
        h_sexo, y_sexo = forward(Wxh_sexo1, Why_sexo1, bh_sexo1, by_sexo1, state)

    rage = int(line[5])
    if rage < 30:
        h_age, y_age = forward(Wxh_age0, Why_age0, bh_age0, by_age0, state)
    elif rage < 40:
        h_age, y_age = forward(Wxh_age1, Why_age1, bh_age1, by_age1, state)
    elif rage < 60:
        h_age, y_age = forward(Wxh_age2, Why_age2, bh_age2, by_age2, state)
    else:
        h_age, y_age = forward(Wxh_age3, Why_age3, bh_age3, by_age3, state)

    new = int(line[7])
    if new==0:
        h_new, y_new = forward(Wxh_new0, Why_new0, bh_new0, by_new0, state)
    else:
        h_new, y_new = forward(Wxh_new1, Why_new1, bh_new1, by_new1, state)

    if line[12]=='A':
        h_rel, y_rel = forward(Wxh_rel0, Why_rel0, bh_rel0, by_rel0, state)
    elif line[12]=='I':
        h_rel, y_rel = forward(Wxh_rel1, Why_rel1, bh_rel1, by_rel1, state)
    elif line[12]=='P':
        h_rel, y_rel = forward(Wxh_rel2, Why_rel2, bh_rel2, by_rel2, state)
    else:
        h_rel, y_rel = forward(Wxh_rel3, Why_rel3, bh_rel3, by_rel3, state)

    title = line[23].split(' ')
    if title[0]=='01':
        h_seg, y_seg = forward(Wxh_seg0, Why_seg0, bh_seg0, by_seg0, state)
    elif title[0]=='02':
        h_seg, y_seg = forward(Wxh_seg1, Why_seg1, bh_seg1, by_seg1, state)
    else:
        h_seg, y_seg = forward(Wxh_seg2, Why_seg2, bh_seg2, by_seg2, state)

    y_raw = np.concatenate((y_sexo, y_age, y_new, y_rel, y_seg), axis=1)
    y_sum = np.dot(y_raw,Wtt)
    output = "%d" % int(line[1])
    for i in range(len(y_sum)):
        if y_sum[i] >-0.002:
            output = "%s,%s" % (output,header[i+24])
    print output
    writer.writerow(output.split(","))

for idx in range(155):
    file = os.listdir('%.3d/' % idx)
    for f in file:
        if f=='.DS_Store': continue
        process("%.3d/%s" % (idx,f))
        print "finish file: %s" % f
