"""
-------------------------------------------------
   File Name：     utils
   Description :
   Author :        lin
   Software:       PyCharm
   date：          2018/11/18 22:33
-------------------------------------------------
   Change Activity:
                   2018/11/18 22:33
-------------------------------------------------
"""
__author__ = 'lin'

'''
TP : 正类判定为正类
TN : 负类判定为负类
FP : 负类判定为正类
FN : 正类判定为负类
'''

def precision(TP,TN,FP,FN):
    return TP / (TP + FP)

def recall(TP,TN,FP,FN):
    return TP / (TP + FN)

def accuracy(TP,TN,FP,FN):
    return (TP + TN) / (TP + TN + FP + FN)

def F1_measure(TP,TN,FP,FN):
    p = precision(TP,TN,FP,FN)
    r = recall(TP,TN,FP,FN)
    return 2 * p * r / (p + r)