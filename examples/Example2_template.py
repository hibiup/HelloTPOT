from tpot import TPOTClassifier


def template_example_1():
    """
    template 参数允许我们为 pipeline 定义一个希望的计算路径以缩减计算时间，并让优化过程更容易理解。例如以下模板为优化定义了以下过程：

    1. 特征选则（Selector）
    2. 特征转换（Transformer）. 虽然在 scikit-leawrn 中 SelectorMixin 过程属于 TransformerMixin 的子过程，但是在这里 Transformer
       并不包含 Selector.
    3. 最后一步是分类（Classifier）运算。(如果选择 TPOTClassifier 那么值必须是 Classifier，同样如果选择 TPOTRegressor，那么值就必须是 Regressor)

    由此得到模板值是： Selector-Transformer-Classifier

    注意：TPOT template 目前只支持现行 linear pipeline。
    """
    pipeline_optimizer = TPOTClassifier(template='Selector-Transformer-Classifier',
                                        generations=5,
                                        population_size=20,
                                        cv=5,
                                        random_state=42,
                                        verbosity=2)

    """
    2) 准备数据集
    """
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                        train_size=0.75, test_size=0.25)

    """
    3) 执行优化器, 寻找最优解
    """
    pipeline_optimizer.fit(X_train, y_train)

    """
    4) 打印结果
    """
    print(pipeline_optimizer.score(X_test, y_test))


    """
    5) 自动生成对应的代码
    """
    pipeline_optimizer.export('../output/example2_pipeline.py')

