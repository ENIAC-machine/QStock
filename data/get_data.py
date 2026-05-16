from moex_api.history import history, trading_listing


tickers= list(set(trading_listing(status='traded')['SECID'].to_list()))

df = history(list(tickers),
             st='2000-01-01',
             end='2026-01-01',
             max_retries=20,
             retry_pause=4,
             verbose=True)

def func(sub_df):
    sub_df = sub_df.T.droplevel(axis=1, level=0)
    # print(sub_df.columns)
    sub_df = sub_df[['TRADEDATE', 'OPEN', 'CLOSE', 'HIGH', 'LOW']]
    # sub_df.set_index('TRADEDATE', inplace=True)
    sub_df.dropna(axis=0, inplace=True)
    return sub_df
df = df.T.groupby(level=0).apply(func).reset_index().\
        rename(columns={'DataFrame' : 'TICKER'}).\
        set_index('TRADEDATE').drop(columns='level_1').\
        sort_values('TRADEDATE')
        
print(df.shape)
df.to_csv('stock_data.csv')
