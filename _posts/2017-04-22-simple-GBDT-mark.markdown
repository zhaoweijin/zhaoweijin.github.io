---
layout: post
title:  "simple GBDT 代码解析!"
date:   2017-04-22 09:18:47 +0800
categories: feature engineering
---

#### [github源码](https://github.com/bluekingsong/simple-gbdt)

**程序流程图如下**

![有帮助的截图]({{ site.url }}/assets/simple-gbdt.png)


## gbt.py



```
if __name__ == "__main__":
   project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
   if project_path not in sys.path:
       sys.path.insert(0, project_path)
       
def main(data_filename, stat_filename, max_iter, sample_rate, learn_rate, max_depth, split_points):
   dataset = DataSet(data_filename); # 初始化数据集
   print "Model parameters configuration:[data_file=%s,stat_file=%s,max_iter=%d,sample_rate=%f,learn_rate=%f,max_depth=%d,split_points=%d]" % (
       data_filename, stat_filename, max_iter, sample_rate, learn_rate, max_depth, split_points);
   dataset.describe();  #区分连续值类型和枚举类型,打印出来
   stat_file = open(stat_filename, "w");
   stat_file.write(
       "iteration\taverage loss in train data\tprediction accuracy on test data\taverage loss in test data\n");
   model = Model(max_iter, sample_rate, learn_rate, max_depth, split_points); # 初始化数据模型
   train_data = sample(dataset.get_instances_idset(), int(dataset.size() * 2.0 / 3.0)); # dataset.get_instances_idset是返回数据集中所有样本的索引id，然后随机2/3个
   test_data = set(dataset.get_instances_idset()) - set(train_data); # 测试数据是全量样本数据-训练数据
   model.train(dataset, train_data, stat_file, test_data); # 数据模型进行训练
   # model.test(dataset,test_data);
   stat_file.close();
   
   
if __name__ == "__main__":

    data_path = sys.path[0]+'/'
    input_filename = data_path+"data/adult.data.csv";
    if len(argv) != 8:
        print "usage:", argv[0], "data_filename stat_filename max_iter sample_rate learn_rate max_depth split_points";
        print "for example:", argv[0], "data/adult.data.csv output/adult.data.stat 50 0.4 0.1 1 -1";
        print "#" * 60;
        print "data_filename: the csv datafile used to train and test( random split 1/3 of data as test part";
        print "stat_filename: the file to hold ouput information about prediction accuracy and loss value in each iteration";
        print "max_iter: set the iterations in gradient boost algorithm";
        print "sample_rate: subsample rate of train data to construct a single decision tree";
        print "learn_rate: the learn rate parameter of gradient boost algorithm";
        print "max_depth: the maximum depth of a decision tree, max_depth=1 means a decision stump with depth=1";
        print "split_points: if a feature is numeric and has many distinct values, it's very slow to find a optimal split point.i use just $split_points$ values to find optimal split point of tree. 0 and negative $split_points$ means do not use the trick";
        print "#" * 60;
    else:
        input_filename = data_path+argv[1]; # 输入数据的文件
        stat_filename = data_path+argv[2]; # 输出结果的文件
        max_iter = int(argv[3]);  # 最大轮询数
        sample_rate = float(argv[4]);  # 采样率
        learn_rate = float(argv[5]);  # 学习率
        max_depth = int(argv[6]);  # 最大深度
        split_points = int(argv[7]);  # 切割点
        main(input_filename, stat_filename, max_iter, sample_rate, learn_rate, max_depth, split_points);   
    
```

## data.py

```
class DataSet:
    def __init__(self,filename):  ## just for csv data format
        line_cnt=0;
        self.instances=dict();
        self.distinct_valueset=dict();  ## just for 数字类型
        for line in open(filename):
            if line=="\n":
                continue;
            fields=line[:-1].split(","); # 第一行为字段数据，之后都是样本数据
            if line_cnt==0:  ## csv head
                self.field_names=tuple(fields);
                # for debug
                # print "field names:",fields;
            else:
                if len(fields)!=len(self.field_names):
                    print "wrong fields:",line;
                    raise ValueError("fields number is wrong!");
                if line_cnt==1: ## determine the value type
                   self.field_type=dict();
                   for i in range(0,len(self.field_names)):
                       valueSet=set();
                       # 使用了一个技巧，如果能转浮点型，就是数字类型实例，否则就是字符串类型实例
                       try:
                           float(fields[i]);
                           self.distinct_valueset[self.field_names[i]]=set();
                       except ValueError:
                           valueSet.add(fields[i]);
                       self.field_type[self.field_names[i]]=valueSet;
                self.instances[line_cnt]=self.__construct_instance__(fields);
            line_cnt=line_cnt+1;
            
    # 构造数据集实例
    def __construct_instance__(self,fields):
        instance=dict();
        for i in range(0,len(fields)):
            field_name=self.field_names[i];
            real_type_mark=self.is_real_type_field(field_name); # 判断是否数字类型字段，返回True/False
            if real_type_mark:
                try:
                    instance[field_name]=float(fields[i]);
                    self.distinct_valueset[field_name].add(float(fields[i]));
                except ValueError:
                    raise ValueError("the value is not float,conflict the value type at first detected");
            else:
                instance[field_name]=fields[i];
                self.field_type[field_name].add(fields[i]);
        return instance;
    def describe(self):
        info="features:"+str(self.field_names)+"\n";
        info=info+"\n dataset size="+str(self.size())+"\n";        
        for field in self.field_names:
            info=info+"description for field:"+field;
            valueset=self.get_distinct_valueset(field);
            if self.is_real_type_field(field):
                info=info+" real value, distinct values number:"+str(len(valueset));
                info=info+" range is ["+str(min(valueset))+","+str(max(valueset))+"]\n";
            else:
                info=info+" enum type, distinct values number:"+str(len(valueset));
                info=info+" valueset="+str(valueset)+"\n";
            info=info+"#"*60+"\n";
        print info;
    def get_instances_idset(self):
        return self.instances.keys();
    def is_real_type_field(self,name):
        if name not in self.field_names:
             raise ValueError(" field name not in the dictionary of dataset");
        # return (name in self.distinct_valueset);
        return len(self.field_type[name])==0;
    def get_label_size(self,name="label"):
        if name not in self.field_names:
            raise ValueError(" there is no class label field!");
        return len(self.field_type[name]);
    def get_label_valueset(self,name="label"):
        if name not in self.field_names:
            raise ValueError(" there is no class label field!");
        return self.field_type[name];
    def size(self):
        return len(self.instances);
    def get_instance(self,Id):
        if not Id in self.instances:
            raise ValueError("Id not in the instances dict of dataset");
        return self.instances[Id];
    def get_attributes(self):
        field_names=[x for x in self.field_names if x!="label"];
        return tuple(field_names);
    def get_distinct_valueset(self,name):
        if not name in self.field_names:
            raise ValueError("the field name not in the dataset field dictionary");
        if self.is_real_type_field(name):
            return self.distinct_valueset[name];
        else:
            return self.field_type[name];

if __name__=="__main__":
    from sys import argv;
    data=DataSet(argv[1]);
    print "instances size=",len(data.instances);
    print data.instances[1];

```





