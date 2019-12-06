## 比赛简介
"合肥高新杯"心电人机智能大赛（https://tianchi.aliyun.com/competition/entrance/231754/introduction）

>依据心电图机8导联的数据，以及病患年龄、性别等因素，用统计学、机器学习、深度学习等方式探索挖掘心电波形与心电异常事件之间的关系，构建精准预测模型。

## 方案
选用2个模型，传统模型和深度模型，加权得到最终结果。
* 传统模型，提取特征，每类分别训练模型
* 深度模型，resnet34

最终复赛成绩[21/2353]

## 文件说明
* dnn 深度模型resnet34代码，使用开源的基线代码，修改了优化器，其它未改动。
* extract_features 传统方法
* submit 复赛docker提交文件

## 评价指标F1
F1为评价指标，结果越大越好，具体计算公式如下：
$$F1 =\frac{2 * P * R}{P+R}$$

> 其中P为准确率，计算公式如下：
P = 预测正确的心电异常事件数 / 预测的心电异常事件数
R为召回率，计算公式如下：
R = 预测正确的心电异常事件数 / 总心电异常事件数
P、R中涉及的心电异常事件数均是所有样本的累加。

## 数据预处理
### 导联数据处理
* 如果大部分导联数据为0，则删除此条记录。
* 数据归一化

### 性别和年龄
* 性别： one-hot编码
* 年龄： 缺失值用平均值代替

### 数据去噪
心电信号是一种随机性强、幅值微弱、信噪比低的生理信号,在采集期间,心电信号极易受到心电图机、电极连接以及病人自身的运动伪迹等因素的影响,从而引入噪声干扰,这些常见的噪声主要包括了肌电噪声、工频噪声和基线漂移等。
#### 方法一 使用过滤器去噪
采样频率为500hz,信号本身最大的频率为250hz，使用带通滤波滤除3hz以下和40hz以上频率成分。
波形信号上有很多噪音，这些噪音都是成百Hz的，这些对于ecg信号波形就属于无用的噪音，我们就噪声信号过滤掉，此时得到的波形就会比较平滑了。
过滤之后会发现R波的峰值会被**削弱**，如果T波的峰值比较大，在接下来工作中可能会造成R波和T波的误判。

#### 方法二 小波降噪
高斯噪声往往由小波域中的小值表示，可以通过将低于给定阈值的系数设置为零（硬阈值）或将所有系数收缩为零（软阈值）来消除。
使用图像的两种不同的小波系数阈值选择方法：Bayesshrink和Visushrink。

>Visushrink算法
Visushrink方法对所有小波细节系数采用一个单一的通用阈值。该阈值用于去除高概率的加性高斯噪声，这种加性高斯噪声往往导致图像过于平滑。通过指定一个小于真实噪声标准偏差的西格玛，可以得到一个更直观的令人满意的结果。

>Bayesshrink算法
Bayesshrink算法是小波软阈值的一种自适应方法，它对每个小波子带估计一个唯一的阈值。这通常会导致对单个阈值所能获得的结果的改进。

采用scikit-image库小波降噪的方法，对源码进行部分修改，对图像三维数据的处理改成二维数据处理。
使用coif5小波基和Visushrink算法，会得到较好的波形，R峰不会被过多削弱。

### 模板提取
一个正常的心电信号在一个周期内，是由P波，QRS波群以及T波组成
>P波：代表心脏除极，一般呈现钝圆形，幅度约为0.25mV，持续时间为0.08~0.11s。
PR间期：指心房除极开始到心室除极开始的传导时间，一般是从P波开始到QRS波的起点，正常PR间隔时间为：0.12s-0.20s。
QRS：代表心室肌的除极产生的电位变化，是心电图中最高大，快速的波形。
T波：代表心室肌的复极
QT间期：代表心脏去极化和复极过程，即心室收缩的总时间。QT 间隔和心率相关，心率越快，QT 间隔越短，反之越长
RR间隔：是指连续两个心电周期 R 峰值间隔变化特征，是常用的心拍识别标准之一。

使用biosppy(Hamilton)提取R波位置，再根据标准模板的pqrst固定时间间隔和RR间隔提取出p波和T波。


## 特征提取
### 时域特征提取
* 均值
> 描述信号的平均水平，即数据的静态分量
* 均方根值
> 描述信号的平均能量或平均功率，是信号能量的最恰当的量度
* 峰值
> 信号在一个周期内的最大值
* 峰值因子
> 峰值因子是信号峰值与均方根值的比值，代表的是峰值在波形中的极端程度。
* 偏度
> 偏度是衡量随机变量概率分布的不对称性，是相对于平均值不对称程度的度量。
* 峭度
> 大幅值非常敏感，当其概率增加时，峭度会迅速增大，这有利于探测奇异振动信号
* 波形因子
> 波形因子定义为均方根值与绝对值之比
* 脉冲因子
> 脉冲因子是信号峰值与整流平均值（绝对值的平均值）的比值
* 裕度因子
> 裕度因子是信号峰值与方根幅值的比值

###  频率特征提取
* fft
* 小波分解能量提取

### hrv特征
* SDNN (Standard Deviation of Normal toNormal)，全部正常心跳间距之标准差，单位为毫秒。
* SDANN (Standard deviation of the averages of NN intervals inall 5-minute segments of the entirerecording)，全程依五分钟分成连续的时段，先计算每五分钟心跳间期的标准差，再计算标准差的平均值，单位为毫秒。
* NN50 count (Number of pairs of adjacent NN intervals differingby more than 50 ms in the entirerecording)，心电图中所有每对相邻正常心跳时间间隔，差距超过50毫秒的数目。
* pNN50 (NN50 count divided by the total number of all NNintervals)，NN50数目除以量测之心电图中所有的正常心跳间隔总数。

## 模型训练
### lightgbm、xgboost、catboost
* xgboost 训练时间太长放弃
* catboost 未调参下，准确率未比lightgbm，由于时间原因放弃
* lightgbm

## 其它的一些想法未实现
fastdtw 采用动态规划来计算两个时间序列之间的相似性，耗时太长，卒。

## 深度模型
使用resnet34模型。尝试过resnet34改进版，得分反而变低。  
优化器 由adam改为Ranger  
针对数据不平衡，loss函数调整  

## 模型融合
可参考[KAGGLE ENSEMBLING GUIDE](https://mlwave.com/kaggle-ensembling-guide/)  
时间关系，只是两个模型加权平均了一下。

## 所使用的库
* pandas
* tqdm
* scikit-learn
* lightgbm==2.1.1
* seaborn
* wfdb
* pywavelets==1.0.3
* pytorch==1.1.0

## 参考代码
<https://physionet.org/challenge/2017/>  
<https://github.com/JavisPeng/ecg_pytorch?spm=5176.12282029.0.0.3d952737ec5tuc>  
<https://github.com/victorkifer/ecg-af-detection-physionet-2017>  
<https://github.com/MLWave/Kaggle-Ensemble-Guide>

