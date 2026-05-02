import pandas as pd
from moex_api.history import history

st='2020-01-01'
end='2021-01-01'

df = history(['YNDX', 'IMOEX'], st='2020-01-01', end='2021-01-01')

cols_to_save = ['TRADEDATE', 'OPEN', 'CLOSE', 'HIGH', 'LOW']
df = df.loc[:, pd.IndexSlice[:, cols_to_save]]
df.columns = df.columns.map('_'.join)

date_cols = [col  for col in df.columns if col.endswith('TRADEDATE')]


df_new = pd.DataFrame({'TRADEDATE' : pd.bdate_range(start=st, end=end)})
for col in date_cols:
    df[col] = pd.to_datetime(df[col])
    stock_cols = [column for column in df.columns if column.startswith(col.split('_')[0])]
    df_new = df_new.merge(df[stock_cols], how='left', left_on='TRADEDATE', right_on=col).drop(columns=[col])

print(df_new.head())
print(df_new.columns)
