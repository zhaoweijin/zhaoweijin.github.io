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
        max_iter = int(argv[3]);  # 最大轮询数(迭代)
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
                self.field_type[field_name].add(fields[i]); # field_type 是字符串类型的字段实例
        return instance;
        
    # 只作为打印输出，对实例不做改变
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
        
    # 获取实例的索引值
    def get_instances_idset(self):
        return self.instances.keys();
        
    # 判断是否是数字类型字段
    def is_real_type_field(self,name):
        if name not in self.field_names:
             raise ValueError(" field name not in the dictionary of dataset");
        # return (name in self.distinct_valueset);
        return len(self.field_type[name])==0;
    # 获取标签的个数
    def get_label_size(self,name="label"):
        if name not in self.field_names:
            raise ValueError(" there is no class label field!");
        return len(self.field_type[name]);
    # 获取标签的值
    def get_label_valueset(self,name="label"):
        if name not in self.field_names:
            raise ValueError(" there is no class label field!");
        return self.field_type[name];
    # 获取数据实例的个数
    def size(self):
        return len(self.instances);
    # 获取某个实例
    def get_instance(self,Id):
        if not Id in self.instances:
            raise ValueError("Id not in the instances dict of dataset");
        return self.instances[Id];
    # 获取数据表头字段
    def get_attributes(self):
        field_names=[x for x in self.field_names if x!="label"];
        return tuple(field_names);
    # 获取字段的值（数字/字符串类型）
    def get_distinct_valueset(self,name):
        if not name in self.field_names:
            raise ValueError("the field name not in the dataset field dictionary");
        if self.is_real_type_field(name):
            return self.distinct_valueset[name]; # 数字类型的值
        else:
            return self.field_type[name]; # 字符串类型的值

if __name__=="__main__":
    from sys import argv;
    data=DataSet(argv[1]); # 数据集实例
    print "instances size=",len(data.instances);
    print data.instances[1];

```

## model.py

```
class Model:
    def __init__(self,max_iter,sample_rate,learn_rate,max_depth,split_points=0):
        self.max_iter=max_iter; # 最大轮询数(迭代)
        self.sample_rate=sample_rate; # 采样率
        self.learn_rate=learn_rate; # 学习率
        self.max_depth=max_depth; # 最大深度
        self.split_points=split_points; # 切割点
        self.trees=dict(); # 决策树
    def train(self,dataset,train_data,stat_file,test_data=None):
       label_valueset=dataset.get_label_valueset(); # 标签的值
       f=dict();  ## for train instances
       self.initialize(f,dataset);  # 初始化样本分类的标签集 e.g. {1:{'>50K':0.0,'<=50K':0.0},..}
       for iter in range(1,self.max_iter+1):
           subset=train_data;
           if self.sample_rate>0 and self.sample_rate<1:
               subset=sample(subset,int(len(subset)*self.sample_rate));  # 抽样训练数据 e.g. train_data*0.4
           self.trees[iter]=dict();
           residual=self.compute_residual(dataset,subset,f); # 计算残差
           #print "resiudal of iteration",iter,"###",residual;
           for label in label_valueset:
               leafNodes=[]; # 叶子节点
               targets={}; # 训练数据的残差值
               for id in subset:
                   targets[id]=residual[id][label];
               ## for debug
               #print "targets of iteration:",iter,"and label=",label,"###",targets;
               tree=construct_decision_tree(dataset,subset,targets,0,leafNodes,self.max_depth,self.split_points);  # 构造决策树
               #if label==sample(label_valueset,1)[0]:
               #    print tree.describe("#"*30+"Tree Description"+"#"*30+"\n");
               self.trees[iter][label]=tree; # 某个迭代的标签子树
               self.update_f_value(f,tree,leafNodes,subset,dataset,label); # 更新训练数据实例的值
           ## for debug
           #print "residual=",residual;
           if test_data!=None:
               accuracy,ave_risk=self.test(dataset,test_data);
           train_loss=self.compute_loss(dataset,train_data,f);
           test_loss=self.compute_loss(dataset,test_data,f);
           stat_file.write(str(iter)+"\t"+str(train_loss)+"\t"+str(accuracy)+"\t"+str(test_loss)+"\n"); 
           if iter%1==0:
               print "accuracy=%f,average train_loss=%f,average test_loss=%f"%(accuracy,train_loss,test_loss);
               label="+";
               print "stop iteration:",iter,"time now:",datetime.now();
               print "\n";
    def compute_loss(self,dataset,subset,f):
        loss=0.0;
        for id in subset:
            instance=dataset.get_instance(id);
            f_values=None;
            if id in f:
                f_values=f[id];
            else:
                f_values=self.compute_instance_f_value(instance,dataset.get_label_valueset());
            exp_values={};
            for label in f_values:
                exp_values[label]=exp(f_values[label]);
            probs={};
            for label in f_values:
                probs[label]=exp_values[label]/sum(exp_values.values());
            loss=loss-log(probs[instance["label"]]);
        return loss/len(subset);
    def initialize(self,f,dataset):
        for id in dataset.get_instances_idset():
            f[id]=dict();
            for label in dataset.get_label_valueset():
                f[id][label]=0.0;
    def update_f_value(self,f,tree,leafNodes,subset,dataset,label):
        data_idset=set(dataset.get_instances_idset());
        subset=set(subset);
        for node in leafNodes:
            for id in node.get_idset():
                f[id][label]=f[id][label]+self.learn_rate*node.get_predict_value();
        ## for id not in subset, we have to predict by retrive the tree
        for id in data_idset-subset:
            f[id][label]=f[id][label]+self.learn_rate*tree.get_predict_value(dataset.get_instance(id));
    def compute_residual(self,dataset,subset,f):
        residual={};
        label_valueset=dataset.get_label_valueset();
        for id in subset:
            residual[id]={};
            p_sum=sum([exp(f[id][x]) for x in label_valueset]);  # 指数e^x 的相加
            for label in label_valueset:
                p=exp(f[id][label])/p_sum;  # 运用Softmax函数,求概率
                y=0.0;
                if dataset.get_instance(id)["label"]==label:
                    y=1.0;
                residual[id][label]=y-p;
        return residual;
    # 计算标签实例的预估值（训练/测试）
    def compute_instance_f_value(self,instance,label_valueset):
        f_value=dict();
        for label in label_valueset:
            f_value[label]=0.0;
        for iter in self.trees: # 轮询树的迭代次数
            for label in label_valueset:
                tree=self.trees[iter][label];
                # 更新标签实例对应的值=标签实例对应的值+学习率*决策树的预估值
                f_value[label]=f_value[label]+self.learn_rate*tree.get_predict_value(instance); # 把测试数据代入决策树分类
        return f_value; # e.g. {' >50K': 0.003975535168195719, ' <=50K': -0.003975535168195719}
    def test(self,dataset,test_data):
        right_predition=0; # 正确预估值
        label_valueset=dataset.get_label_valueset(); # 获取标签值
        risk=0.0; # 总预估错的风险
        for id in test_data:
            instance=dataset.get_instance(id); #获取实例样本值
            predict_label,probs=self.predict_label(instance,label_valueset); # 预估的标签，预估正确的几率
            single_risk=0.0; # 单个样本预估错的几率
            for label in probs:
                if label==instance["label"]:
                    single_risk=single_risk+(1.0-probs[label]); # 1-成功几率
                else:
                    single_risk=single_risk+probs[label]; # 预估错的几率
            #print probs,"instance label=",instance["label"],"##single_risk=",single_risk/len(probs);
            risk=risk+single_risk/len(probs); #总预估值+单预估几率/预估次数
            if instance["label"]==predict_label:
                right_predition=right_predition+1;
        #print "test data size=%d,test accuracy=%f"%(len(test_data),float(right_predition)/len(test_data));       
        return float(right_predition)/len(test_data),risk/len(test_data);
    
    # 预估标签
    def predict_label(self,instance,label_valueset):
        # 计算实例的标签值
        f_value=self.compute_instance_f_value(instance,label_valueset);
        predict_label=None;
        exp_values=dict();
        for label in f_value:
            # e^实例的标签值（e^x）
            exp_values[label]=exp(f_value[label]);
        exp_sum=sum(exp_values.values());
        probs=dict();
        for label in exp_values:
            probs[label]=exp_values[label]/exp_sum; # softmax函数
        for label in probs:
            if predict_label==None or probs[label]>probs[predict_label]:
                predict_label=label;
       return predict_label,probs; # 返回预估标签和预估的标签softmax值
```

## tree.py
```
class Tree:
    def __init__(self):
        self.split_feature=None;
        self.leftTree=None;
        self.rightTree=None;
        ## condition for real value is < , for category value is =
        ## is for the left-path tree
        self.real_value_feature=True;
        self.conditionValue=None;
        self.leafNode=None;
    # 获取预估值
    def get_predict_value(self,instance):
        # 是否叶节点
        if self.leafNode!=None:  ## we are in the leaf node
            return self.leafNode.get_predict_value(); # 返回叶节点（Leafnode）预估值
        # 切分的特征
        if self.split_feature==None:
            raise ValueError("the tree is null");
        # 是否是数字类型，并且小于当前的条件值
        if self.real_value_feature and instance[self.split_feature]<self.conditionValue:
            return self.leftTree.get_predict_value(instance);
        elif not self.real_value_feature and instance[self.split_feature]==self.conditionValue:
            return self.leftTree.get_predict_value(instance);
        return self.rightTree.get_predict_value(instance);
    def describe(self,addtion_info=""):
        if self.leftTree==None or self.rightTree==None:
            return self.leafNode.describe();
        leftInfo=self.leftTree.describe();
        rightInfo=self.rightTree.describe();
        info=addtion_info+"{split_feature:"+str(self.split_feature)+",split_value:"+str(self.conditionValue)+"[left_tree:"+leftInfo+",right_tree:"+rightInfo+"]}";
        return info;
class LeafNode:
    def __init__(self,idset):
        self.idset=idset;
        self.predictValue=None;
    def describe(self):
        return "{LeafNode:"+str(self.predictValue)+"}";
    def get_idset(self):
        return self.idset;
    def get_predict_value(self):
        return self.predictValue;
    def update_predict_value(self,targets,K): ## K is number of class, just for classification
        sum1=sum([targets[x] for x in self.idset]);
        sum2=sum([abs(targets[x])*(1.0-abs(targets[x])) for x in self.idset]);
#        self.predictValue=float(K-1)/K*(sum1/sum2);
        #self.predictValue=sum([targets[x] for x in self.idset])/len(self.idset);
        if sum1==0:
            self.predictValue=0;
        else:
            try:
                self.predictValue=float(K-1)/K*(sum1/sum2);
            except ZeroDivisionError:
                print "zero division,sum1=%f,sum2=%f"%(sum1,sum2);
                print "targets are:",[targets[x] for x in self.idset];
                raise;
        ## for debug
        #print "targets=",[targets[x] for x in self.idset];
        #print "sum1=",sum1,"sum2=",sum2;
        #print "predict value=",self.predictValue; 
def compute_min_loss(values):
    if len(values)<2:
        return 0;
    mean=sum(values)/float(len(values));
    loss=0.0;
    for v in values:
        loss=loss+(mean-v)*(mean-v);
    return loss;
## if split_points is larger than 0, we just random choice split_points to evalute minLoss when consider real-value split
def construct_decision_tree(dataset,remainedSet,targets,depth,leafNodes,max_depth,split_points=0):
    #print "start process,depth=",depth;
    if depth<max_depth:
        attributes=dataset.get_attributes();
        loss=-1;
        selectedAttribute=None;
        conditionValue=None;
        selectedLeftIdSet=[];
        selectedRightIdSet=[];       
        for attribute in attributes:
           # print "start process attribute=",attribute;
            is_real_type=dataset.is_real_type_field(attribute);
            attrValues=dataset.get_distinct_valueset(attribute);  # 获取全量样本的某个字段的值
            if is_real_type and split_points>0 and len(attrValues)>split_points:  ## need subsample split points to speed up
                attrValues=sample(attrValues,split_points);
            for attrValue in attrValues:
                leftIdSet=[];
                rightIdSet=[];
                for Id in remainedSet:
                    instance=dataset.get_instance(Id);
                    value=instance[attribute];
                    if (is_real_type and value<attrValue)or(not is_real_type and value==attrValue):   ## fall into the left
                        leftIdSet.append(Id);
                    else:
                        rightIdSet.append(Id);
                leftTargets=[targets[id] for id in leftIdSet];
                rightTargets=[targets[id] for id in rightIdSet];
                sumLoss=compute_min_loss(leftTargets)+compute_min_loss(rightTargets);
                if loss<0 or sumLoss<loss:
                    selectedAttribute=attribute;
                    conditionValue=attrValue;
                    loss=sumLoss;
                    selectedLeftIdSet=leftIdSet;
                    selectedRightIdSet=rightIdSet;
            #print "for attribute:",attribute," min loss=",loss;
       # print "process over, get split attribute=",selectedAttribute;
        if selectedAttribute==None or loss<0:
            raise ValueError("cannot determine the split attribute.");      
        tree=Tree();
        tree.split_feature=selectedAttribute;
        tree.real_value_feature=dataset.is_real_type_field(selectedAttribute);
        tree.conditionValue=conditionValue;
        tree.leftTree=construct_decision_tree(dataset,selectedLeftIdSet,targets,depth+1,leafNodes,max_depth);
        tree.rightTree=construct_decision_tree(dataset,selectedRightIdSet,targets,depth+1,leafNodes,max_depth);
        #print "build a tree,min loss=",loss,"conditon value=",conditionValue,"attribute=",tree.split_feature;
        return tree;
    else:  ## is a leaf node
        node=LeafNode(remainedSet);
        K=dataset.get_label_size();
        node.update_predict_value(targets,K);
        leafNodes.append(node); ## add a leaf node
        tree=Tree();
        tree.leafNode=node;
        return tree;
```





