"""
NN for Santander competitation at Kaggle
Written by Aasta Lin
BSD License
"""

import numpy as np
import pickle
import csv
import os

num = 5
fsize = 24
osize = 24
learning_rate = 0.01
Wtt = np.random.randn(num,1)*0.2

# sexo NN
hsize_sexo = 100
Wxh_sexo0 = np.random.randn(hsize_sexo, fsize)*0.01
Wxh_sexo1 = np.random.randn(hsize_sexo, fsize)*0.01
Why_sexo0 = np.random.randn(osize, hsize_sexo)*0.01
Why_sexo1 = np.random.randn(osize, hsize_sexo)*0.01
bh_sexo0 = np.zeros((hsize_sexo, 1))
bh_sexo1 = np.zeros((hsize_sexo, 1))
by_sexo0 = np.zeros((osize, 1))
by_sexo1 = np.zeros((osize, 1))

mWxh_sexo0,  mWhy_sexo0 = np.zeros_like(Wxh_sexo0), np.zeros_like(Why_sexo0)
mWxh_sexo1,  mWhy_sexo1 = np.zeros_like(Wxh_sexo1), np.zeros_like(Why_sexo1)
mbh_sexo0, mby_sexo0 = np.zeros_like(bh_sexo0), np.zeros_like(by_sexo0)
mbh_sexo1, mby_sexo1 = np.zeros_like(bh_sexo1), np.zeros_like(by_sexo1)


# age NN
hsize_age = 20
Wxh_age0 = np.random.randn(hsize_age, fsize)*0.01
Wxh_age1 = np.random.randn(hsize_age, fsize)*0.01
Wxh_age2 = np.random.randn(hsize_age, fsize)*0.01
Wxh_age3 = np.random.randn(hsize_age, fsize)*0.01
Why_age0 = np.random.randn(osize, hsize_age)*0.01
Why_age1 = np.random.randn(osize, hsize_age)*0.01
Why_age2 = np.random.randn(osize, hsize_age)*0.01
Why_age3 = np.random.randn(osize, hsize_age)*0.01
bh_age0 = np.zeros((hsize_age, 1))
bh_age1 = np.zeros((hsize_age, 1))
bh_age2 = np.zeros((hsize_age, 1))
bh_age3 = np.zeros((hsize_age, 1))
by_age0 = np.zeros((osize, 1))
by_age1 = np.zeros((osize, 1))
by_age2 = np.zeros((osize, 1))
by_age3 = np.zeros((osize, 1))

mWxh_age0,  mWhy_age0 = np.zeros_like(Wxh_age0), np.zeros_like(Why_age0)
mWxh_age1,  mWhy_age1 = np.zeros_like(Wxh_age1), np.zeros_like(Why_age1)
mWxh_age2,  mWhy_age2 = np.zeros_like(Wxh_age2), np.zeros_like(Why_age2)
mWxh_age3,  mWhy_age3 = np.zeros_like(Wxh_age3), np.zeros_like(Why_age3)
mbh_age0, mby_age0 = np.zeros_like(bh_age0), np.zeros_like(by_age0)
mbh_age1, mby_age1 = np.zeros_like(bh_age1), np.zeros_like(by_age1)
mbh_age2, mby_age2 = np.zeros_like(bh_age2), np.zeros_like(by_age2)
mbh_age3, mby_age3 = np.zeros_like(bh_age3), np.zeros_like(by_age3)


# new NN
hsize_new = 50
Wxh_new0 = np.random.randn(hsize_new, fsize)*0.01
Wxh_new1 = np.random.randn(hsize_new, fsize)*0.01
Why_new0 = np.random.randn(osize, hsize_new)*0.01
Why_new1 = np.random.randn(osize, hsize_new)*0.01
bh_new0 = np.zeros((hsize_new, 1))
bh_new1 = np.zeros((hsize_new, 1))
by_new0 = np.zeros((osize, 1))
by_new1 = np.zeros((osize, 1))

mWxh_new0,  mWhy_new0 = np.zeros_like(Wxh_new0), np.zeros_like(Why_new0)
mWxh_new1,  mWhy_new1 = np.zeros_like(Wxh_new1), np.zeros_like(Why_new1)
mbh_new0, mby_new0 = np.zeros_like(bh_new0), np.zeros_like(by_new0)
mbh_new1, mby_new1 = np.zeros_like(bh_new1), np.zeros_like(by_new1)


# rel NN
hsize_rel = 20
Wxh_rel0 = np.random.randn(hsize_rel, fsize)*0.01
Wxh_rel1 = np.random.randn(hsize_rel, fsize)*0.01
Wxh_rel2 = np.random.randn(hsize_rel, fsize)*0.01
Wxh_rel3 = np.random.randn(hsize_rel, fsize)*0.01
Why_rel0 = np.random.randn(osize, hsize_rel)*0.01
Why_rel1 = np.random.randn(osize, hsize_rel)*0.01
Why_rel2 = np.random.randn(osize, hsize_rel)*0.01
Why_rel3 = np.random.randn(osize, hsize_rel)*0.01
bh_rel0 = np.zeros((hsize_rel, 1))
bh_rel1 = np.zeros((hsize_rel, 1))
bh_rel2 = np.zeros((hsize_rel, 1))
bh_rel3 = np.zeros((hsize_rel, 1))
by_rel0 = np.zeros((osize, 1))
by_rel1 = np.zeros((osize, 1))
by_rel2 = np.zeros((osize, 1))
by_rel3 = np.zeros((osize, 1))

mWxh_rel0,  mWhy_rel0 = np.zeros_like(Wxh_rel0), np.zeros_like(Why_rel0)
mWxh_rel1,  mWhy_rel1 = np.zeros_like(Wxh_rel1), np.zeros_like(Why_rel1)
mWxh_rel2,  mWhy_rel2 = np.zeros_like(Wxh_rel2), np.zeros_like(Why_rel2)
mWxh_rel3,  mWhy_rel3 = np.zeros_like(Wxh_rel3), np.zeros_like(Why_rel3)
mbh_rel0, mby_rel0 = np.zeros_like(bh_rel0), np.zeros_like(by_rel0)
mbh_rel1, mby_rel1 = np.zeros_like(bh_rel1), np.zeros_like(by_rel1)
mbh_rel2, mby_rel2 = np.zeros_like(bh_rel2), np.zeros_like(by_rel2)
mbh_rel3, mby_rel3 = np.zeros_like(bh_rel3), np.zeros_like(by_rel3)


# seg NN
hsize_seg = 50
Wxh_seg0 = np.random.randn(hsize_seg, fsize)*0.01
Wxh_seg1 = np.random.randn(hsize_seg, fsize)*0.01
Wxh_seg2 = np.random.randn(hsize_seg, fsize)*0.01
Why_seg0 = np.random.randn(osize, hsize_seg)*0.01
Why_seg1 = np.random.randn(osize, hsize_seg)*0.01
Why_seg2 = np.random.randn(osize, hsize_seg)*0.01
bh_seg0 = np.zeros((hsize_seg, 1))
bh_seg1 = np.zeros((hsize_seg, 1))
bh_seg2 = np.zeros((hsize_seg, 1))
by_seg0 = np.zeros((osize, 1))
by_seg1 = np.zeros((osize, 1))
by_seg2 = np.zeros((osize, 1))

mWxh_seg0,  mWhy_seg0 = np.zeros_like(Wxh_seg0), np.zeros_like(Why_seg0)
mWxh_seg1,  mWhy_seg1 = np.zeros_like(Wxh_seg1), np.zeros_like(Why_seg1)
mWxh_seg2,  mWhy_seg2 = np.zeros_like(Wxh_seg2), np.zeros_like(Why_seg2)
mbh_seg0, mby_seg0 = np.zeros_like(bh_seg0), np.zeros_like(by_seg0)
mbh_seg1, mby_seg1 = np.zeros_like(bh_seg1), np.zeros_like(by_seg1)
mbh_seg2, mby_seg2 = np.zeros_like(bh_seg2), np.zeros_like(by_seg2)


def forward(Wxh, Why, bh, by, inputs):
    hs = np.tanh(np.dot(Wxh, inputs) + bh)
    ys = np.dot(Why,hs) + by
    return hs, ys


def backward(dy, hs, ys, Wxh, Why, bh, by, inputs):
    dWhy = np.dot(dy, hs.T)
    dby = dy
    dh = np.dot(Why.T, dy)
    dhraw = (1 - hs * hs) * dh
    dbh = dhraw
    dWxh = np.dot(dhraw, inputs.T)
    for dparam in [dWxh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)
    return dWxh, dWhy, dbh, dby


def update(Wxh, Why, bh, by, dWxh, dWhy, dbh, dby, mWxh, mWhy, mbh, mby):
    for param, dparam, mem in zip([Wxh, Why, bh, by], 
                                  [dWxh, dWhy, dbh, dby], 
                                  [mWxh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)


def process(filename):
    ptr = open(filename,"r")
    reader = csv.reader(ptr)
    header = next(reader)

    # first column
    line = next(reader)
    for i in range(len(line)): 
        if line[i]=='NA':
            return 0

    tmp =  np.array(line[24:49], dtype='Float32')
    state = np.reshape(tmp.astype(np.float), (fsize,1))

    loss = 0
    while True:
        try:
            line = next(reader)
            tmp =  np.array(line[24:49], dtype='Float32')
            data = np.reshape(tmp.astype(np.float), (fsize,1))
            target = np.maximum(0, data-state)

            # forward pass
            if line[4]=='V':
                sexo = 0
                h_sexo, y_sexo = forward(Wxh_sexo0, Why_sexo0, bh_sexo0, by_sexo0, state)
            else:
                sexo = 1
                h_sexo, y_sexo = forward(Wxh_sexo1, Why_sexo1, bh_sexo1, by_sexo1, state)

            rage = int(line[5])
            if rage < 30:
                age = 0
                h_age, y_age = forward(Wxh_age0, Why_age0, bh_age0, by_age0, state)
            elif rage < 40:
                age = 1
                h_age, y_age = forward(Wxh_age1, Why_age1, bh_age1, by_age1, state)
            elif rage < 60:
                age = 2
                h_age, y_age = forward(Wxh_age2, Why_age2, bh_age2, by_age2, state)
            else:
                age = 3
                h_age, y_age = forward(Wxh_age3, Why_age3, bh_age3, by_age3, state)

            new = int(line[7])
            if new==0:
                h_new, y_new = forward(Wxh_new0, Why_new0, bh_new0, by_new0, state)
            else:
                h_new, y_new = forward(Wxh_new1, Why_new1, bh_new1, by_new1, state)

            type = int(line[11])-1
            if line[12]=='A':
                rel = 0
                h_rel, y_rel = forward(Wxh_rel0, Why_rel0, bh_rel0, by_rel0, state)
            elif line[12]=='I':
                rel = 1
                h_rel, y_rel = forward(Wxh_rel1, Why_rel1, bh_rel1, by_rel1, state)
            elif line[12]=='P':
                rel = 2
                h_rel, y_rel = forward(Wxh_rel2, Why_rel2, bh_rel2, by_rel2, state)
            else:
                rel = 3
                h_rel, y_rel = forward(Wxh_rel3, Why_rel3, bh_rel3, by_rel3, state)

            title = line[23].split(' ')
            if title[0]=='01':
                seg = 0
                h_seg, y_seg = forward(Wxh_seg0, Why_seg0, bh_seg0, by_seg0, state)
            elif title[0]=='02':
                seg = 1
                h_seg, y_seg = forward(Wxh_seg1, Why_seg1, bh_seg1, by_seg1, state)
            else:
                seg = 2
                h_seg, y_seg = forward(Wxh_seg2, Why_seg2, bh_seg2, by_seg2, state)
            y_raw = np.concatenate((y_sexo, y_age, y_new, y_rel, y_seg), axis=1)

            # calculate loss
            y_sum = np.dot(y_raw,Wtt)
            y = np.maximum(0, y_sum)
            dy = np.copy(y)
            dy = dy-target
            for i in range(len(y_sum)):
                if y_sum[i]<0:
                    dy[i] = 0
           
            for i in range(len(target)):
                if target[i]==0:
                    loss += np.maximum(0,y[i]-0.5)
                else:
                    loss += np.maximum(0,0.5-y[i])

            # backward pass
            dy_raw = np.dot(dy, Wtt.T)

            dy_sexo = np.reshape(dy_raw[:,0], (osize,1))
            if sexo==0:
                dWxh, dWhy, dbh, dby = backward(dy_sexo, h_sexo, y_sexo,
                                       Wxh_sexo0, Why_sexo0, bh_sexo0, by_sexo0, state)
                update(Wxh_sexo0, Why_sexo0, bh_sexo0, by_sexo0,
                       dWxh, dWhy, dbh, dby, mWxh_sexo0, mWhy_sexo0, mbh_sexo0, mby_sexo0)
            else:
                dWxh, dWhy, dbh, dby = backward(dy_sexo, h_sexo, y_sexo,
                                       Wxh_sexo1, Why_sexo1, bh_sexo1, by_sexo1, state)
                update(Wxh_sexo1, Why_sexo1, bh_sexo1, by_sexo1,
                       dWxh, dWhy, dbh, dby, mWxh_sexo1, mWhy_sexo1, mbh_sexo1, mby_sexo1)

            dy_age = np.reshape(dy_raw[:,1], (osize,1))
            if age==0:
                dWxh, dWhy, dbh, dby = backward(dy_age, h_age, y_age,
                                       Wxh_age0, Why_age0, bh_age0, by_age0, state)
                update(Wxh_age0, Why_age0, bh_age0, by_age0,
                       dWxh, dWhy, dbh, dby, mWxh_age0, mWhy_age0, mbh_age0, mby_age0)
            elif age==1:
                dWxh, dWhy, dbh, dby = backward(dy_age, h_age, y_age,
                                       Wxh_age1, Why_age1, bh_age1, by_age1, state)
                update(Wxh_age1, Why_age1, bh_age1, by_age1,
                       dWxh, dWhy, dbh, dby, mWxh_age1, mWhy_age1, mbh_age1, mby_age1)
            elif age==2:
                dWxh, dWhy, dbh, dby = backward(dy_age, h_age, y_age,
                                       Wxh_age2, Why_age2, bh_age2, by_age2, state)
                update(Wxh_age2, Why_age2, bh_age2, by_age2,
                       dWxh, dWhy, dbh, dby, mWxh_age2, mWhy_age2, mbh_age2, mby_age2)
            else:
                dWxh, dWhy, dbh, dby = backward(dy_age, h_age, y_age,
                                       Wxh_age3, Why_age3, bh_age3, by_age3, state)
                update(Wxh_age3, Why_age3, bh_age3, by_age3,
                       dWxh, dWhy, dbh, dby, mWxh_age3, mWhy_age3, mbh_age3, mby_age3)

            dy_new = np.reshape(dy_raw[:,2], (osize,1))
            if new==0:
                dWxh, dWhy, dbh, dby = backward(dy_new, h_new, y_new,
                                       Wxh_new0, Why_new0, bh_new0, by_new0, state)
                update(Wxh_new0, Why_new0, bh_new0, by_new0,
                       dWxh, dWhy, dbh, dby, mWxh_new0, mWhy_new0, mbh_new0, mby_new0)
            else:
                dWxh, dWhy, dbh, dby = backward(dy_new, h_new, y_new,
                                       Wxh_new1, Why_new1, bh_new1, by_new1, state)
                update(Wxh_new1, Why_new1, bh_new1, by_new1,
                       dWxh, dWhy, dbh, dby, mWxh_new1, mWhy_new1, mbh_new1, mby_new1)

            dy_rel = np.reshape(dy_raw[:,3], (osize,1))
            if rel==0:
                dWxh, dWhy, dbh, dby = backward(dy_rel, h_rel, y_rel,
                                       Wxh_rel0, Why_rel0, bh_rel0, by_rel0, state)
                update(Wxh_rel0, Why_rel0, bh_rel0, by_rel0,
                       dWxh, dWhy, dbh, dby, mWxh_rel0, mWhy_rel0, mbh_rel0, mby_rel0)
            elif rel==1:
                dWxh, dWhy, dbh, dby = backward(dy_rel, h_rel, y_rel,
                                       Wxh_rel1, Why_rel1, bh_rel1, by_rel1, state)
                update(Wxh_rel1, Why_rel1, bh_rel1, by_rel1,
                       dWxh, dWhy, dbh, dby, mWxh_rel1, mWhy_rel1, mbh_rel1, mby_rel1)
            elif rel==2:
                dWxh, dWhy, dbh, dby = backward(dy_rel, h_rel, y_rel,
                                       Wxh_rel2, Why_rel2, bh_rel2, by_rel2, state)
                update(Wxh_rel2, Why_rel2, bh_rel2, by_rel2,
                       dWxh, dWhy, dbh, dby, mWxh_rel2, mWhy_rel2, mbh_rel2, mby_rel2)
            else:
                dWxh, dWhy, dbh, dby = backward(dy_rel, h_rel, y_rel,
                                       Wxh_rel3, Why_rel3, bh_rel3, by_rel3, state)
                update(Wxh_rel3, Why_rel3, bh_rel3, by_rel3,
                       dWxh, dWhy, dbh, dby, mWxh_rel3, mWhy_rel3, mbh_rel3, mby_rels3)

            dy_seg = np.reshape(dy_raw[:,4], (osize,1))
            if seg==0:
                dWxh, dWhy, dbh, dby = backward(dy_seg, h_seg, y_seg,
                                       Wxh_seg0, Why_seg0, bh_seg0, by_seg0, state)
                update(Wxh_seg0, Why_seg0, bh_seg0, by_seg0,
                       dWxh, dWhy, dbh, dby, mWxh_seg0, mWhy_seg0, mbh_seg0, mby_seg0)
            elif seg==1:
                dWxh, dWhy, dbh, dby = backward(dy_seg, h_seg, y_seg,
                                       Wxh_seg1, Why_seg1, bh_seg1, by_seg1, state)
                update(Wxh_seg1, Why_seg1, bh_seg1, by_seg1,
                       dWxh, dWhy, dbh, dby, mWxh_seg1, mWhy_seg1, mbh_seg1, mby_seg1)
            else:
                dWxh, dWhy, dbh, dby = backward(dy_seg, h_seg, y_seg,
                                       Wxh_seg2, Why_seg2, bh_seg2, by_seg2, state)
                update(Wxh_seg2, Why_seg2, bh_seg2, by_seg2,
                       dWxh, dWhy, dbh, dby, mWxh_seg2, mWhy_seg2, mbh_seg2, mby_seg2)
        except ValueError:
            break
        except StopIteration:
            break
        state = data
    return loss

for epoch in range(90):
    for idx in range(155):
        file = os.listdir('%.3d/' % idx)
        for f in file:
            if f=='.DS_Store': continue
            loss = process("%.3d/%s" % (idx,f))
            print "[epoch %d]%s %f" % (epoch,f,loss)
    if (epoch+1)%10==0:
        with open('weight1_%d.pickle' % (epoch+1), 'w') as ff:
          pickle.dump([Wtt,
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
                       Wxh_seg2, Why_seg2, bh_seg2, by_seg2], ff)
