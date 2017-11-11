
# coding: utf-8

# # 加载数据

# In[568]:


### 加载数据集
import pandas as pd
df =pd.read_csv('titanic_data.csv')
df.head(10)


# # 数据清洗和整理

# 针对第一步加载数据，对数据的查看，可看到PassengerId 、Name 、 Ticket 都是每一行的唯一值，对于统计数据无意义，可以对这些列予以废弃

# In[569]:


#http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.drop.html
df=df.drop(['PassengerId','Name','Ticket'],axis=1)
df.head(10)


# In[570]:


print df.info()
print '='*50
print df.describe()


# 从上图的df.info()结果可知，Age、Cabin和Embarked存在缺失值，而Cabin只有204条非空值，可废弃此维度，不作为统计依据
# 
# 另外，由于Age和Embarked的缺失值可接受范围，但是要进行空值填充，统计时需体现出这些数
# 
# 从上面可知Age是数值型，Embaked是字符串类型，而Age的最小值为0.42,大于0，所以此处空值可以填充为-1
# 
# 

# In[571]:


df=df.drop('Cabin' ,axis=1)
df=df.fillna(-1)
print df.info()
print '==============================='
df.head(10)


# In[572]:


get_ipython().magic(u'pylab inline')
#查看总体直方图
df.hist()


# ## 数据探索

# In[573]:


#先看看整体的生还率
df.Survived.describe()
all_count=df.Survived.count()
#总体生还率
survived_count=df.Survived[df.Survived==1].count()
survived_rate=survived_count*1.0/all_count
survived_rate


# 总体生还率：38%

# In[574]:


df.columns


# Embarked/Sex是字符类型，是离散变量，
# 
# 上面的直方图统计是针对数值型的，所以可以针对上面直方图的除了Survived列的值进行查看，以区分出连续变量和离散变量（甚至判断连续变量是否可作为离散变量对待）
# 
# 约定枚举值大于10个的数值型，视为连续变量处理
# 数值型列：Age/Fare/Parch/Pclass/SibSp

# In[575]:


def getEnumCount(col_name):
    return len(df.groupby(col_name).groups.keys())


# In[576]:


num_cols=['Age','Fare','Parch','Pclass','SibSp']
for i in num_cols:
    print i,getEnumCount(i)


# 由此可看到，Age/Fare作为连续型变量
# 
# Parch/Pclass/SibSp视作离散变量处理，加上Embarked/Sex 有：
# 
# 连续变量
# 
#  - continuous_cols=['Age','Fare']
# 
# 离散变量
# 
#  - discrete_cols=['Parch','Pclass','SibSp','Embarked','Sex']

# In[577]:


continuous_cols=['Age','Fare']
discrete_cols=['Parch','Pclass','SibSp','Embarked','Sex']


# # 离散变量分组统计
# 
# Survived作为指标，其他作为<font color=red >维度</font>，依次对各维度与指标的关系进行查看
# 
# 把维度值都作为离散变量进行分析

# 为了方便统计，加入一列extra_col用于统计数量时作为固定列进行索引取数

# In[578]:


df['extra_col']=1
df.head(10)


# 针对离散变量，编写分类统计函数，统计各个类别的生还率

# In[579]:


def survived_rate_by(df,col_name):
    t=df.groupby(by=[col_name,'Survived']).sum()['extra_col']
    df1=pd.DataFrame(t)
    df2=df1.unstack()
    return df2


# In[580]:


for i in discrete_cols:
    df2=survived_rate_by(df,i)
    df2.plot(kind='bar',title='survived/unsurvived num of %s' % (i,))


# 上图分别按各离散变量分组后，不能获救的人数域获救的人数柱形图，情况分别如下：
# 
# 从kaggle(https://www.kaggle.com/c/titanic/data)数据集的说明中 可以看到各列的意思：
# 
# ## Data Dictionary
# - Variable	Definition	Key
# - survival	Survival	0 = No, 1 = Yes
# - pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
# - sex	Sex	
# - Age	Age in years	
# - sibsp	# of siblings / spouses aboard the Titanic	
# - parch	# of parents / children aboard the Titanic	
# - ticket	Ticket number	
# - fare	Passenger fare	
# - cabin	Cabin number	
# - embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton
# 
# ## Variable Notes
# 
# **pclass**: A proxy for socio-economic status (SES)
#  - 1st = Upper
#  - 2nd = Middle
#  - 3rd = Lower
# 
# age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# **sibsp** : The dataset defines family relations in this way...  
# Sibling = brother, sister, stepbrother, stepsister  
# Spouse = husband, wife (mistresses and fiancés were ignored)  
# 
# **parch**: The dataset defines family relations in this way...  
# Parent = mother, father  
# Child = daughter, son, stepdaughter, stepson  
# Some children travelled only with a nanny, therefore parch=0 for them.

# ### 上图小结1
# 
# 上面各个图说明的情况是：
# 
# **Parch**  
#  - 可看到独身旅游的人最多，但是独身旅游的人相对 其他有父母孩子的人 的死亡人数比获救人数多  
# 
# **Pclass**  
# Pclass代表经济地位（1的经济地位最高）  
#  - 社会地位低的人数最多  
#  - 社会地位低的死亡人数比其他两个等级的死亡人数多很多  
#  - 社会地位低的生存人数比其他两个等级的生存人数不低地位高的多很多  
# 
# **SibSp**  
#  - 无兄弟姐妹的人最多
# 
# **Embarked**
#  - 在Southampton上船的人最多  
# 
# **Sex**
#  - 女性中：  生存人数是死亡人数的两倍多  
#  - 男性中：  死亡人数是生存人数的3-4倍

# ### 计算各个离散变量对应的生还率

# In[581]:


#获取生还率
def get_Survived_rate(df,col_name):
    df2=pd.DataFrame(survived_rate_by(df,col_name))
    df2.columns=df2.columns.droplevel() #http://pandas.pydata.org/pandas-docs/stable/generated/pandas.MultiIndex.droplevel.html
    df2['Survived_rate']=df2[1]/(df2[0]+df2[1])
    df2=df2.fillna(0)
    return df2
def  showplot(df,col_name):
    df2=get_Survived_rate(df,col_name)
    print df2
    df2.Survived_rate.plot(kind='bar' ,title='Survived Rate of %s' % (col_name,))
    df2.drop('Survived_rate',axis=1).plot(kind='bar' ,title='Survived  of %s' % (col_name,))
    
#隔分类与总人数占比
all_count=df.count()['Pclass']
def count_rate(tmp_df):
    tmp_df['people_rate']=(tmp_df[0]+tmp_df[1])/all_count
    return tmp_df


# ### Parch

# In[582]:


showplot(df,'Parch')


# In[583]:


count_rate(get_Survived_rate(df,'Parch'))


# 总体平均的生还率是38%  
# 从这里可看到高于总体生还率的是有1-3个的父母/子女的生还率高于无父母/孩子的，Parch=3及以上的人数都是个位数，剩下的Parch=1/2的生还率高于总体生还率  
# 

# ### Pclass

# In[584]:


showplot(df,'Pclass')
count_rate(get_Survived_rate(df,'Pclass'))


# 这里可看到，社会等级越高，生还率越高

# ### SibSp 

# In[585]:


showplot(df,'SibSp')
count_rate(get_Survived_rate(df,'SibSp'))


# 这里可看到有1/2兄弟姐妹的部分生还率比总体的高

# In[586]:


showplot(df,'Embarked')
count_rate(get_Survived_rate(df,'Embarked'))


# Cherbourg登船的生还率最高

# ### Sex

# In[587]:


showplot(df,'Sex')
count_rate(get_Survived_rate(df,'Sex'))


# 可看到女星的生还率比男性的生还率高出三倍多

# ## 对连续变量处理
# continuous_cols=['Age','Fare']
# 
# ### Age

# In[588]:


#去掉无法确认的值
age_df=df[df['Age']>-1]
age_df=age_df.drop(['Pclass','Sex','SibSp','Fare','Embarked','Parch','extra_col'],axis=1)
age_df.head(10)


# In[589]:


age_df.mean()


# 剔去无法确认年龄的数据后，总体的生还率是40.6%

# In[590]:


#死亡的
age_df['dead']=1-age_df['Survived']
age_df.head(10)


# In[591]:


age_df.groupby(by='Age').sum().plot()


# 上图蓝色线代表生还人数，橙色线代表死亡人数  
# 可看到10岁以下，蓝色线普遍位于橙色线之上，代表生还率高于50%；  
# 但是，从15岁到30岁这一个年龄段，橙色线明显高于蓝色线，生还率比较低

# In[592]:


agg_age_df=age_df.groupby(by='Age').sum()
agg_age_df['Survived_reate']=agg_age_df['Survived']/(agg_age_df['Survived']+agg_age_df['dead'])
agg_age_df.Survived_reate.plot(title='Survived Rate of Age')


# 可看道10岁以下生化率大于总体生还率，20-60岁的生还率都是统一范围内波动，都小于60%

# ### Fare

# In[593]:


#去掉无法确认的值
fare_df=df.drop(['Pclass','Sex','SibSp','Age','Embarked','Parch','extra_col'],axis=1)
fare_df.head(10)


# In[594]:


#生存的
fare_df['dead']=1-fare_df['Survived']
fare_df.head(10)


# In[595]:


fare_df.groupby(by='Fare').sum().plot()


# 由于票价都是集中在100以内,截取100内看看

# In[596]:


fare_df=fare_df[fare_df['Fare']<100]
fare_df.info()


# In[597]:


fare_df.groupby(by='Fare').sum().plot()


# 票价与生还率之间无太大关系
