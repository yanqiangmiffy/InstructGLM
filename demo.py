#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: demo.py
@time: 2023/03/24
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""

def predict(a=1,b=2,**kwargs):
    print(a,b)
    print(kwargs) #{'c': 3} #{}
    print({**kwargs})
    print({'e':'f',**kwargs}) # {'e': 'f', 'c': 3, 'd': 2}

print(predict(a=1,b=2,c=3,d=2))