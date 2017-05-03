---
layout: post
title:  "owenzhang _2b_generate_dataset_for_vw_fm 代码解析!"
date:   2017-05-02 10:18:47 +0800
categories: feature engineering
---

#### [github源码](https://github.com/owenzhang/kaggle-avazu)


## _2b_generate_dataset_for_vw_fm.py



```
#coding:utf-8
import pandas as pd
import numpy as np
import scipy as sc
import scipy.sparse as sp
from sklearn.utils import check_random_state 
import pylab 
import sys
import time
import utils
from utils import *
import os

from sklearn.externals.joblib import dump, load


# 导入_1_中的dat
t0 = load('tmp_data/t0.joblib_dat')
print "t0 loaded with shape", t0.shape

# dev_id_cnt_2: dev_id_cnt 对应的是一个dev_id的在数据集出现的个数，若t0['dev_id_cnt2']>300替换其值=300,划分区间
# dev_ip_cnt2: 同上
t0['dev_id_cnt2'] = np.minimum(t0.cnt_dev_id.astype('int32').values, 300)
t0['dev_ip_cnt2'] = np.minimum(t0.cnt_dev_ip.astype('int32').values, 300)

# 目的:筛出值为1的情况，独立作为一个值，区分在id和ip在频数=1时的有无情况不同表现权重
# dev_id2plus: 复制t0['dev_id2plus']其值，若一个dev_id的在数据集出现次数=1,其值替换='___only1'
# dev_ip2plus: 同上
t0['dev_id2plus'] = t0.device_id.values
t0.ix[t0.cnt_dev_id.values == 1, 'dev_id2plus'] = '___only1'
t0['dev_ip2plus'] = t0.device_ip.values
t0.ix[t0.cnt_dev_ip.values == 1, 'dev_ip2plus'] = '___only1'

# 利用时间维度看dev_id在不同天同次出现的情况
# 一个dev_id的在数据集出现的个数
t0['device_ip_only_hour_for_day'] = t0.cnt_device_ip_day_hour.values == t0.cnt_device_ip_pday.values

# 目的:得到app_site_id+X的分布情况
# 对app_site_id[app_id + site_id]和以下feature进行合并(主要是类别feature)
vns0 = ['app_or_web', 'banner_pos', 'C1', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
for vn in vns0 + ['C14']:
    print vn
    vn2 = '_A_' + vn
    t0[vn2] = np.add(t0['app_site_id'].values, t0[vn].astype('string').values)
    t0[vn2] = t0[vn2].astype('category')

t3 = t0
vns1 = vns0 + ['hour1'] + ['_A_' + vn for vn in vns0] + \
 ['device_model', 'device_type', 'device_conn_type', 'app_site_id', 'as_domain', 'as_category',
      'cnt_device_ip_day_hour', 'cnt_device_ip_day_hour_prev', 'cnt_device_ip_day_hour_next', 'cnt_device_ip_pday',
     'cnt_diff_device_ip_day_pday', 'as_model'] + \
    [ 'dev_id_cnt2', 'dev_ip_cnt2', 'C14', '_A_C14', 'dev_ip2plus', 'dev_id2plus']
 
#'cnt_device_ip_day', 'device_ip_only_hour_for_day'
    
t3a = t3.ix[:, ['click']].copy()
idx_base = 3000
for vn in vns1:
    if vn in ['cnt_device_ip_day_hour', 'cnt_device_ip_day_hour_prev', 'cnt_device_ip_day_hour_next', 'cnt_device_ip_pday',
     'cnt_diff_device_ip_day_pday', 'cnt_device_ip_day', 'cnt_device_ip_pday']:

        # 对时间频数进行划区间，即x = t3[vn].values，x在[-100,200],>200,x=200,<-100,x=-100
        _cat = pd.Series(np.maximum(-100, np.minimum(200, t3[vn].values))).astype('category').values.codes
    elif vn in ['as_domain']:

        # 目的:得到app_domain+site_domain的分布情况
        _cat = pd.Series(np.add(t3['app_domain'].values, t3['site_domain'].values)).astype('category').values.codes
    elif vn in ['as_category']:

        # 目的:得到app_category+site_category的分布情况

        _cat = pd.Series(np.add(t3['app_category'].values, t3['site_category'].values)).astype('category').values.codes
    elif vn in ['as_model']:

        # 目的:得到app_site_id+device_model的分布情况

        _cat = pd.Series(np.add(t3['app_site_id'].values, t3['device_model'].values)).astype('category').values.codes
    else:
        # 得到此feature其值的分布情况
        _cat = t3[vn].astype('category').values.codes

    _cat = np.asarray(_cat, dtype='int32')
    _cat1 = _cat + idx_base
    # 全量数据+3000,弱化序列数据的影响
    t3a[vn] = _cat1
    print vn, idx_base, _cat1.min(), _cat1.max(), np.unique(_cat).size

    # 对单个feature的最大值进行累计,+1为了防止_cat.max()=0
    idx_base += _cat.max() + 1


print "to save t3a ..."
t3a_save = {}
# t3a = 经过数据转换后的t3a
t3a_save['t3a'] = t3a
# 可跟_2c_的id_base做校对
t3a_save['idx_base'] = idx_base
dump(t3a_save, 'tmp_data/t3a.joblib_dat')

    
```




