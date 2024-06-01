import pandas as pd

df_feature = pd.read_parquet('../Dataset/bank_marketing_features.gzip')
df_label = pd.read_parquet('../Dataset/bank_marketing_label.gzip')


df_feature['label'] = (df_label['y'] == 'yes')
df_feature['education'] = df_feature['education'].replace([None, 'secondary', 'primary', 'tertiary'], [0, 1, 2, 3])

print(df_feature.corr(method='pearson', numeric_only=True))


