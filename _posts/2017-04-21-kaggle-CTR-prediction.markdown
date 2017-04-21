---
layout: post
title:  "kaggle CTR prediction!"
date:   2017-04-21 17:18:47 +0800
categories: jekyll update
---

# kaggle CTR prediction

## [Avazu](https://www.kaggle.com/c/avazu-ctr-prediction)

#### Data介绍

**File descriptions**

* train.gz - Training set. 10 days of click-through data, ordered chronologically. Non-clicks and clicks are subsampled according to different strategies. 10天的click-through数据，按时间先后顺序排列。未点击和点击数据根据不同策略二次抽样。
* test.gz - Test set. 1 day of ads to for testing your model predictions. 1天的广告数据用来测试模型预测结果。
* sampleSubmission.gz - Sample submission file in the correct format, corresponds to the All-0.5 Benchmark. 提交文件样例。

**Data fields**

* id: ad identifier 广告标示
* click: 0/1 for non-click/click 点击／未点击
* hour: format is YYMMDDHH, so 14091123 means 23:00 on Sept. 11, 2014 UTC. 时间戳
* C1 -- anonymized categorical variable 匿名类别变量
* banner_pos 广告位置
* site_id 站点id
* site_domain 站点域
* site_category 站点类别
* app_id 应用id
* app_domain 应用域
* app_category 应用类别
* device_id 设备id
* device_ip 设备ip
* device_model 设备模型
* device_type 设备类型
* device_conn_type 设备连接方式
* C14-C21 -- anonymized categorical variables 匿名类别变量


数据样例：24项特征

| id                   | click | hour     | C1   | banner_pos | site_id  | site_domain | site_category | app_id   | app_domain | app_category | device_id | device_ip | device_model | device_type | device_conn_type | C14   | C15  | C16  | C17  | C18  | C19  | C20    | C21  |
| -------------------- | ----- | -------- | ---- | ---------- | -------- | ----------- | ------------- | -------- | ---------- | ------------ | --------- | --------- | ------------ | ----------- | ---------------- | ----- | ---- | ---- | ---- | ---- | ---- | ------ | ---- |
| 1000009418151094273  | 0     | 14102100 | 1005 | 0          | 1fbe01fe | f3845767    | 28905ebd      | ecad2386 | 7801e8d9   | 07d7df22     | a99f214a  | ddd2926e  | 44956a24     | 1           | 2                | 15706 | 320  | 50   | 1722 | 0    | 35   | -1     | 79   |
| 10000169349117863715 | 0     | 14102100 | 1005 | 0          | 1fbe01fe | f3845767    | 28905ebd      | ecad2386 | 7801e8d9   | 07d7df22     | a99f214a  | 96809ac8  | 711ee120     | 1           | 0                | 15704 | 320  | 50   | 1722 | 0    | 35   | 100084 | 79   |
| 10000371904215119486 | 0     | 14102100 | 1005 | 0          | 1fbe01fe | f3845767    | 28905ebd      | ecad2386 | 7801e8d9   | 07d7df22     | a99f214a  | b3cf8def  | 8a4875bd     | 1           | 0                | 15704 | 320  | 50   | 1722 | 0    | 35   | 100084 | 79   |



第一名：https://github.com/guestwalk/kaggle-avazu

第二名：https://github.com/owenzhang/kaggle-avazu

第三名：http://techblog.youdao.com/?p=1211

Avazu Click-Through Rate Prediction是移动广告dsp公司avazu在kaggle上举办的广告点击率预测的比赛，全球共有1604支队伍参加，我和周骁聪组队参加了比赛，最终我们获得了第三名。下面是我们比赛的方法分享。

[Avazu CTR Prediction](http://techblog.youdao.com/wp-content/uploads/2015/03/Avazu-CTR-Prediction.pdf)

整个比赛历时两个多月，在这个过程中我们和众多数据挖掘高手交流切磋，对于广告点击率预测问题也有了更深的理解。点击率预测算法在经历了多年的研究和发展之后，如今再要有一个大的提升已经是一个比较困难的事情了。当前点击率预测算法的主要收益大都是来自于线性模型的改进+特征工程，最近几年工业界已经开始尝试用非线性模型去解决点击率预测的问题，也取得了一些不错的结果。通过这次比赛，我们也相信非线性模型的使用以及多模型的融合会带来点击率预测算法的下一次飞跃。



**owenzhang方法**

* 组合模型
  * RandomForest/sklearn, 
  * GBDT/xgboost,
  * OnlineSGD/Vowpal Wabbit, 
  * Factorization machine/3 idiots
* 特征选择与提取
  * 组合基于site/app的特征，因为这些特征是互补的，组合节省空间
  * prior day mean(y)编码为分类特征，对单变量和多变量都如此
  * device_ip的counts/sequences，为用户身份的代理
  * FM预测使用原始特征和counts/sequences
  * app_site_id * C14-21
* 模型训练
  * f
* 代码描述
  * ​





FTRL  Practical Lessons from Predicting Clicks on Ads at Facebook ADKDD14

GBDT

FM http://www.algo.uni-konstanz.de/members/rendle/pdf/Rendle2010FM.pdf

FFM http://tech.meituan.com/deep-understanding-of-ffm-principles-and-practices.html

http://www.csie.ntu.edu.tw/~r01922136/slides/ffm.pdf

libffm：https://github.com/guestwalk/libffm

Ad click prediction: a view from the trenches

3 idiots' Approach for Display Advertising Challenge

---


## [Criteo](https://www.kaggle.com/c/criteo-display-ad-challenge)

#### Data介绍

**File descriptions**

* **train.csv** - The training set consists of a portion of Criteo's traffic over a period of 7 days. Each row corresponds to a display ad served by Criteo. Positive (clicked) and negatives (non-clicked) examples have both been subsampled at different rates in order to reduce the dataset size. The examples are chronologically ordered. 训练集包含7天的Criteo流量。正样本和负样本做二次采样减少数据集大小。样本按时间先后顺序排列。
* **test.csv** - The test set is computed in the same way as the training set but for events on the day following the training period. 测试集为紧接训练集时间的一天记录。
* **random_submission.csv** - A sample submission file in the correct format.

**Data fields**

* **Label** - Target variable that indicates if an ad was clicked (1) or not (0). 点击/未点击
* **I1-I13** - A total of 13 columns of integer features (mostly count features). 13个整数特征
* **C1-C26** - A total of 26 columns of categorical features. The values of these features have been hashed onto 32 bits for anonymization purposes. 26个类别特征，值被hash为32bit做匿名处理。

The semantic of the features is undisclosed. When a value is missing, the field is empty. 缺失值为空。



https://github.com/guestwalk/kaggle-2014-criteo

