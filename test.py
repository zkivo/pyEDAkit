from pyEDAkit import standardization as std
import pandas as pd

iris_df = pd.read_csv("data/iris.data")

iris_df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
