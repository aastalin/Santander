# Santander Product Recommendation
Implement a simple nn for kaggle competitions:
</br>
</br>
Santander Product Recommendation:
</br>
<https://www.kaggle.com/c/santander-product-recommendation>

---
##0. Preprocess
Generate folder from 001 to 155
```C
$ python
$ import os
$ for i in range(155):
$     os.makedirs("%.3d" % i)
$
$ exit()
```
</br>
Use following script to separate data by user ID.
```C
$ python parser.py
```
</br>
##1. Simple NN by python script
**Train:**
```
$ python integrateNN_1.py # larger hiddle node
$ python integrateNN_2.py # smaller hiddle node
```
**Predict:**
```
$ python predict.py
```
</br>
##2. Concept
Use five small nn to model different type user, including
```
sexo
age
new
relation
segment
```
Then integrate five nn with trainable weighted sum
</br>
```
Y = w0*Ysexo + w1*Yage + w2*Ynew + w3*Yrel + w4*Yseg
```
</br>
</br>
</br>
</br>
**Feel free to download and make it your use-case : )**
