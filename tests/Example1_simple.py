from tpot import TPOTClassifier

"""
1) 生成机器学习 pipeline 优化器. 优化器的作用就是用来寻找最优机器学习 pipeline.

参数说明：
  generations: 执行优化处理的迭代次数．次数越多，效果越好，但是耗时也越长．(缺省是100)
  population_size: 每一个 generation 中的＂人口＂数．数量越大，效果越好．(缺省是100)
  offspring_size: 子孙数量．每一个 generation 最终保留的子孙数量,也是越大越好.
    TPOT 运算消耗的时间和以上三个参数的关系是: 耗时 = POPULATION_SIZE + GENERATIONS x OFFSPRING_SIZE
  cv: 每一次迭代使用的 k-fold cross-validation 的数量.
  random_state: 随机数种子.(通过可以每次都给定同一个值来 re-produce 运算.)
"""
pipeline_optimizer = TPOTClassifier(generations=5,
                                    population_size=20,
                                    cv=5,
                                    random_state=42,
                                    verbosity=2)

"""
2) 准备数据集. 以 iris 数据集为例:
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                    train_size=0.75, test_size=0.25)

"""
3) 执行优化器, 寻找最优解

执行的过程会得到类似以下输出：
===
Optimization Progress:  29%|██▉       | 35/120 [00:05<00:20,  4.18pipeline/s]Generation 1 - Current best internal CV score: 0.9643939393939395
Optimization Progress:  46%|████▌     | 55/120 [00:06<00:10,  6.23pipeline/s]Generation 2 - Current best internal CV score: 0.9643939393939395
Optimization Progress:  63%|██████▎   | 76/120 [00:09<00:06,  6.76pipeline/s]Generation 3 - Current best internal CV score: 0.9643939393939395
Optimization Progress:  81%|████████  | 97/120 [00:10<00:02, 10.66pipeline/s]Generation 4 - Current best internal CV score: 0.9734848484848484
Optimization Progress:  93%|█████████▎| 112/120 [00:13<00:01,  4.34pipeline/s]Generation 5 - Current best internal CV score: 0.9734848484848484
===
"""
pipeline_optimizer.fit(X_train, y_train)

"""
4) 打印结果

得到类是以下输出：
Best pipeline: LogisticRegression(PolynomialFeatures(input_matrix, degree=2, include_bias=False, interaction_only=False), C=10.0, dual=False, penalty=l2)
1.0

这表明 LogisticRegression 算法在 PolynomialFeatures 特征集的情况下,测得了最优 CV
"""
print(pipeline_optimizer.score(X_test, y_test))


"""
5) 自动生成对应的代码

TPOT 一旦得出算法和特征集, 即可自动生成与之对应的代码, 在 export 出得代码中可以看到：
===
  training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)
         
  exported_pipeline = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    LogisticRegression(C=10.0, dual=False, penalty="l2")
  )
  
  exported_pipeline.fit(training_features, training_target)
===

make_pipeline 方法根据自动测出的参数来生成数据集和pipeline。可以在这份输出的代码之上修改出最终的代码。

"""
pipeline_optimizer.export('../output/example1_pipeline.py')
