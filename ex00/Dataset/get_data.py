from ucimlrepo import fetch_ucirepo
import pandas as pd

# fetch dataset
bank_marketing = fetch_ucirepo(id=222)

# data (as pandas data frames)
X = pd.DataFrame(bank_marketing.data.features)
X.to_parquet('bank_marketing_features.gzip', engine="pyarrow", compression='gzip')
X.to_csv('./bank_marketing_features.csv')

y = pd.DataFrame(bank_marketing.data.targets)
y.to_parquet('bank_marketing_label.gzip', engine="pyarrow", compression='gzip')
y.to_csv('./bank_marketing_label.csv')

# metadata 
print(X)
  
# variable information 
#print(bank_marketing.variables)