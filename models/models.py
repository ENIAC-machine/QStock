import os
os.environ["USE_TF"] = "NO"
os.environ["USE_TORCH"] = "YES"

from warnings import filterwarnings

filterwarnings('ignore')


import sys
import pickle
import inspect
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

from itertools import islice
from corus import load_lenta, load_lenta2, load_mokoron, load_buriy_news, load_buriy_webhose
from torch.utils.data import DataLoader, ConcatDataset, Dataset

from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification, 
        PatchTSTConfig, PatchTSTForPrediction
        )

from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from moex_api.history import history, trading_listing

from pathlib import Path
from typing import Callable, Generator, Any, Iterable, Sequence, Annotated, Literal
from abc import abstractmethod, ABC

NEWS_DATA_DIR = os.path.join('..', 'data')

#global variable to map preprocessing functions to the files
func_to_data = {load_lenta: ['lenta-ru-news.csv.gz'],
                load_lenta2 : ['lesnta-ru-news.cv.bz2'],
                load_mokoron : ['db.sql'],
                load_buriy_news : ['news-articles-2014.tar.bz2',
                                   'news-articles-2015-part1.tar.bz2',
                                   'news-articles-2015-part2.tar.bz2'],
                load_buriy_webhose : ['webhose-2016.tar.bz2'],
                pd.read_csv : [file for file in os.listdir(NEWS_DATA_DIR) if file.endswith('.csv')]
                }




class Abstract_Fin_Dataset(Dataset, ABC):

    def __init__(self,
                 data_dir: str | Path,
                 batch_size: int = 32,
                 unified_filenm: str = 'all_data.csv',
                 load_from_file: bool = False,
                 delete_old: bool = True,
                 ) -> None:

        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_elements = 0
        self.unified_filenm = unified_filenm
        self.unified_filepath = os.path.join(data_dir, unified_filenm)
        self.delete_old = delete_old
        
        if not load_from_file:
            self.load()
        else:
            self.num_elements = sum(len(chunk) for chunk in pd.read_csv(self.unified_filepath, chunksize=10_000))
            self.columns = next(pd.read_csv(self.unified_filepath, chunksize=1)).columns
            self.shape = (self.num_elements, len(self.columns))

    def get_init_args(self, local_vars: dict) -> dict:
        kwargs = local_vars.copy()
        kwargs.pop('self', None)
        kwargs.pop('__class__', None)
        self.__dict__.update(kwargs)
        init_vars = set(
                [param.name for param in 
                 inspect.signature(Abstract_Fin_Dataset.__init__).parameters.values() 
                 if param.name != 'self']
                )
        #print(init_vars)
        kwargs = {key : val for key, val in kwargs.items() if key in init_vars}
        return kwargs

    @abstractmethod
    def load(self, *args, **kwargs) -> None:
        '''
        Abstract function to load the dataset into a file

        Inputs:
            -

        Outputs:
            -
        '''
    
        ...
        return None

    #TODO: rewrite to polars
    def __add__(self,
                other
                ) -> pd.DataFrame:
        
        df_mine = pd.read_csv(self.unified_filenm)
        df_other = pd.read_csv(other.unified_filenm)

        df_res = df_mine.merge(df_other, on='date', how='left')

        return df_res


    def __len__(self) -> int:
        return self.num_elements

    def __getitem__(self, idx: int) -> list[Any]:
        return torch.Tensor(pd.read_csv(self.unified_filepath,
                                        skiprows=idx,
                                        nrows=1
                                        ).iloc[0, 2:].to_numpy().astype(float)
                            )

class Joint_Dataset(Abstract_Fin_Dataset):

    def __init__(self,
                 data_dir: str | Path = os.path.join('..', 'data'),
                 batch_size: int = 32,
                 slice_size: int = 10_000,
                 unified_filenm: str = 'all_data.csv',
                 load_from_file: bool = False,
                 delete_old: bool = True,
                 lookback: int = 10,
                 horizon: int = 10
                 ) -> None:

        self.slice_size = slice_size
        local_vars = locals()
        local_vars.pop('slice_size')

        #self.get_init_args(local_vars)
        super().__init__(**self.get_init_args(local_vars))
    
        #fills during the .to_sentiment method
        self.sentiment_filepath: str = None

        return None

    def _prepare_data(self,
                      load_func: Callable,
                      data_path: str | Path = None,
                      verbose: bool = True
                      ):# -> Generator[pd.DataFrame | None]:

        '''
        Function to prepare data of a single file (archive of csv) into a pd.DataFrame object

        Inputs:
            data_path: str | Path - path to the file
            load_func: Callable - function to load it with
            slice_size: int = 10_000 - size of a chunk to process at a time
            verbose: bool = True - verbosity flag

        Outputs:
            pd.DataFrame - chunk of the file

        '''

        end = data_path.split('.')[-1]
        #print(end)

        #print(f'data_path={data_path}')
        try:
            match end:

                case 'csv':

                    reader = load_func(data_path, chunksize=self.slice_size)
                    for df in tqdm(reader,
                                   desc='Loading DataFrame chunks',
                                   leave=False,
                                   disable=not verbose,
                                   unit=' chunks'):
                        self.num_elements += df.shape[0]
                        yield df


                case 'gz' | 'bz2' | 'sql':

                    #make the object an iterable for the `islice` function to work
                    gen = iter(load_func(data_path))
                    
                    lines_cnt = 0
                    with tqdm(unit=' lines', leave=False, disable=not verbose) as progress:
                        while True: #because we don't know the num of elements in the generator
                            data = list(islice(gen, self.slice_size))
                            
                            if data is None or len(data) == 0:
                                break

                            print(data[-1].__attributes__)

                            columns = ['date', 'text']
                             
                            df = pd.DataFrame(data, columns=data[-1].__attributes__)
                            
                            if 'timestamp' in df.columns:
                                df['date'] = pd.to_datetime(df['timestamp']).apply(lambda x:
                                                                                   x.date()
                                                                                   )
                                print('!')


                            if pd.unique(df['date'])[0] == None:

                                #attempt to reconstruct date from url, drop values that can't be converted 
                                df['date'] = pd.to_datetime(
                                                df['url'].apply(lambda x:'/'.join(
                                                    x.split('news/')[-1].split('/')[:3])
                                                                ),
                                                errors='coerce'
                                                ).dropna(how='any',
                                                axis=0
                                                ).reset_index(drop=True).apply(lambda x:x.date())
                            df = df[columns]
                        
                            yield df 
                            
                            lines_cnt += df.shape[0]
                            self.num_elements += df.shape[0]
                            progress.update(self.slice_size) 

                case _:
                    if verbose:
                        print(f'Unknown datatype to process: {data_path.split(".")}')
                    yield None

        except:
            if verbose:
                print(f'Load failed for file {data_path} with loading function {load_func}')
            yield None

    def to_sentiment(self,
                     new_filepath: str | None = None,
                     mdl_cfg: dict | Any | None = None,
                     batch_size: int = 100,
                     verbose: bool = True
                     )->None:
        
        if new_filepath is None:
            new_filepath = Path(self.unified_filepath).with_name('all_data_sentiment.csv')
        
        self.sentiment_filepath = new_filepath

        mdl = Sentiment_Model(cfg=mdl_cfg)
        mdl.eval()

        reader = pd.read_csv(self.unified_filepath, chunksize=batch_size) 
        pd.DataFrame(columns=['date', 'sentiment']).to_csv(new_filepath, header=True)
        
        with torch.no_grad():
       
            for df in tqdm(reader,
                           desc='Converting text to sentiment',
                           total=int(np.ceil(self.num_elements / batch_size)),
                           leave=False,
                           disable=not verbose,
                           unit=' batches'):
                
                #(batch_size, 1)
                sentiment_scores = mdl.forward(
                                df['text'].astype(str).fillna('nothing').tolist()
                                )

                df['sentiment'] = sentiment_scores.cpu().numpy()
                
                df.to_csv(new_filepath,
                          header=False,
                          mode='a',
                          columns=['date', 'sentiment']
                          )
            return None

    #TODO: make the case for low memory, or just rewrite to polars
    def agg(self,
            agg_func: str | Callable = 'mean',
            verbose: bool = False
            ) -> pd.DataFrame:
        ''' 
        Aggregates the data for the sentiment
        '''

        #here I reset and drop 'index' cause the index col may be 'date' or smth
        df = pd.read_csv(self.sentiment_filepath).reset_index(drop=False).drop(columns='index')

        #just in case there will be unnamed cols due to indices and stuff
        df.drop(columns=[col for col in df.columns if 'Unnamed' in col])

        agg_sentiment_path = Path(self.sentiment_filepath).with_name('agg_sentiment.csv')
        if verbose:
            print('Aggregating data...')
        df_new = df.groupby(by='date').mean()
        if verbose:
            print('Data aggregated, converting into .csv format...')

        df_new.to_csv(agg_sentiment_path)

        #reset the path to fetch data
        self.unified_filenm =  agg_sentiment_path
        self.num_elements = len(df_new)
        return df_new


    def load(self,
             sentiment_cfg: dict = {
                 "model_name": "ProsusAI/finbert",
                 "num_labels": 3,                  
                 "device": "cuda" if torch.cuda.is_available() else 'cpu',
                 "score" : True
                 },
             sentiment_batch_size: int = 100,
             st: str = '2014-01-01',
             end: str = '2026-01-01',
             tickers: list[str] | tuple[str] = ['GAZP', 'YNDX', 'NVTK', 'SBER', 'VTBR', 'LKOH',
                                                'GMKN', 'NLMK', 'MGNT', 'AFKS', 'AFLT', 'MTSS',
                                                'HYDR', 'FEES', 'ALRS', 'PLZL', 'CHMF', 'MAGN',
                                                'MOEX', 'TATN', 'SNGS'], 
             verbose: bool = True
             ) -> None:

        '''
        Opens archives and saves their contents in a single file

        Inputs:
            self.data_dir:str | Path = NEWS_DATA_DIR - inputs in the format (function_to_process_file, file_names)
            self.slice_size: int = 10_000 - size of a chunk DataFrame to load from each archive
            self.delete_old: bool = True - flag to delete the old file
            self.unified_filepath: str = 'all_data.csv' - name of the new unified file
            verbose: bool = True - verbosity flag

        Outputs:
            None

        '''

        if os.path.exists(self.unified_filepath) and self.delete_old:
            if verbose:
                print('Found file with the same name in the data directory, deleting...')
            os.remove(self.unified_filepath)
        

        pd.DataFrame(columns=['date', 'text']).to_csv(self.unified_filepath,
                                                      header=True)
        if os.path.exists('checkpoint.pkl'):
            with open('checkpoint.pkl', 'rb') as file:
                checkpoint: dict[str, int] = pickle.load(file)
        else:
            checkpoint = {'load_func' : 0,
                          'data_path' : 0,
                          'df' : 0,
                          'dt' : 0
                          }

        def save_checkpoint() -> None:
            with open('checkpoint.pkl', 'wb') as file:
                pickle.dump(checkpoint, file)
            return None

        total_nans = 0

        for load_func, data_paths in tqdm(func_to_data.items(),
                                          desc='Loading data',
                                          unit=' file batches',
                                          colour='green',
                                          disable=not verbose):

            #print(data_paths)
            if self.unified_filenm in data_paths:
                data_paths.remove(self.unified_filenm)

            for idx, filenm in tqdm(enumerate(data_paths),
                                    desc='Adding paths',
                                    leave=False,
                                    disable=not verbose):

                data_paths[idx] = os.path.join(self.data_dir, filenm) #integrate full path with filename
           
            if len(data_paths):
                data_path = data_paths[0]
            else:
                continue


            for data_path in tqdm(data_paths,
                                  desc=f'Loading archive {data_path}',
                                  leave=False,
                                  unit=" files",
                                  disable=not verbose,
                                  initial=checkpoint['data_path']):

                for df in tqdm(self._prepare_data(data_path = data_path,
                                                  load_func=load_func,
                                                  verbose=verbose),
                               unit= " batches",
                               leave=False,
                               disable=not verbose#,
                               #initial=checkpoint['df']
                               ):

                    if df is None:
                        continue

                    print(df.head())

                    total_nans += len(df[df["date"].isna()])
                    
                    df = df.dropna(subset='date')
                    '''
                    print(f'{data_path}\n\n{df}')
                    ans = input('break?')

                    if ans == 'yes':
                        break
                    '''

                    df.to_csv(self.unified_filepath,
                             header=False,
                              mode='a',
                              columns=['date', 'text']
                              )
                else:
                    continue
                break

            else:
                continue
            break

        print(f'Total nans found: {total_nans}')
        
        self.to_sentiment(batch_size=sentiment_batch_size, mdl_cfg=sentiment_cfg)
        df_sentiment: pd.DataFrame = self.agg(verbose=True)

        if verbose:
            print('Finished loading sentiment data, starting to load the stock data...')

        tickers = tickers[:2]

        df = history(list(tickers),
             st=st,
             end=end,
             max_retries=20,
             retry_pause=10,
             verbose=True)

        def func(sub_df):
            sub_df = sub_df.T.droplevel(axis=1, level=0)
            sub_df = sub_df[['TRADEDATE', 'OPEN', 'CLOSE', 'HIGH', 'LOW']]
            sub_df.dropna(axis=0, inplace=True)
            return sub_df
        
        cols_to_save = ['TRADEDATE', 'OPEN', 'CLOSE', 'HIGH', 'LOW']
        df = df.loc[:, pd.IndexSlice[:, cols_to_save]]
        df.columns = df.columns.map('_'.join)

        date_cols = [col  for col in df.columns if col.endswith('TRADEDATE')]

        df_new = pd.DataFrame({'TRADEDATE' : pd.bdate_range(start=st,
                                                            end=end)
                               }
                              )
        for col in date_cols:
            df[col] = pd.to_datetime(df[col])
            stock_cols = [column for column in df.columns
                          if column.startswith( col.split('_')[0] )
                          ]
            df_new = df_new.merge(df[stock_cols],
                                  how='left',
                                  left_on='TRADEDATE',
                                  right_on=col
                                  ).drop(columns=[col])

        df_stock = df_new.dropna(axis=0, how='any').reset_index(drop=True)
        print(df_new.shape)
        print(df_new.head())
        self.num_elements = len(df)
        df_stock.to_csv(self.unified_filenm)
        self.columns = df_stock.columns.to_list()
        self.num_elements = len(range(len(df_stock) - self.lookback - self.horizon + 1))
        self.shape: tuple = (self.num_elements, df_stock.shape[1])
       
        df_sentiment['date'] = pd.to_datetime(df_sentiment['date'])
        df_stock['TRADEDATE'] = pd.to_datetime(df_stock['TRADEDATE'])

        df_res = df_sentiment.merge(df_stock,
                                    left_on='date', right_on='TRADEDATE',
                                    how='inner').drop(columns='date')
    

        self.shape = df_res.shape
        self.num_elements = df_res.shape[0] - self.lookback - self.horizon + 1
        self.df = df_res
        df_res.to_csv(self.unified_filepath.with_name('preprocessed_data.csv'),
                      index=False)
        return df_res


    def __getitem__(self,
                    idx: int
                    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        
        real_idx = self.lookback + idx

        #reshaped to (batch, 1)
        sentiment = torch.Tensor(list(self.df.loc[real_idx, 'sentiment']).to_numpy().\
                                                                            reshape(-1, 1).\
                                                                            astype(float)
                                 )

        x = self.df.iloc[real_idx - self.lookback: real_idx, :].drop(columns=['TRADEDATE',
                                                                                'sentiment']
                                                                       ).to_numpy().astype(float)

        y = self.df.iloc[real_idx: real_idx + self.horizon, :].drop(columns=['TRADEDATE',
                                                                              'sentiment']
                                                                     ).to_numpy().astype(float)

        
        x, y = torch.Tensor(x),torch.Tensor(y)

        return sentiment, x, y

class News_Dataset(Abstract_Fin_Dataset):

    def __init__(self,
                 data_dir: str | Path = os.path.join('..', 'data'),
                 batch_size: int = 32,
                 slice_size: int = 10_000,
                 unified_filenm: str = 'all_data.csv',
                 load_from_file: bool = False,
                 delete_old: bool = True,
                 ) -> None:

        self.slice_size = slice_size
        local_vars = locals()
        local_vars.pop('slice_size')

        #self.get_init_args(local_vars)
        super().__init__(**self.get_init_args(local_vars))
    
        #fills during the .to_sentiment method
        self.sentiment_filepath: str = None

    def _prepare_data(self,
                      load_func: Callable,
                      data_path: str | Path | None = None,
                      verbose: bool = True
                      ):# -> Generator[pd.DataFrame | None]:

        '''
        Function to prepare data of a single file (archive of csv) into a pd.DataFrame object

        Inputs:
            data_path: str | Path - path to the file
            load_func: Callable - function to load it with
            slice_size: int = 10_000 - size of a chunk to process at a time
            verbose: bool = True - verbosity flag

        Outputs:
            pd.DataFrame - chunk of the file

        '''

        end = data_path.split('.')[-1]
        #print(end)

        #print(f'data_path={data_path}')
        try:
            match end:

                case 'csv':

                    reader = load_func(data_path, chunksize=self.slice_size)
                    for df in tqdm(reader,
                                   desc='Loading DataFrame chunks',
                                   leave=False,
                                   disable=not verbose,
                                   unit=' chunks'):
                        self.num_elements += df.shape[0]
                        yield df


                case 'gz' | 'bz2' | 'sql':

                    #make the object an iterable for the `islice` function to work
                    gen = iter(load_func(data_path))
                    
                    lines_cnt = 0
                    with tqdm(unit=' lines', leave=False, disable=not verbose) as progress:
                        while True: #because we don't know the num of elements in the generator
                            data = list(islice(gen, self.slice_size))
                            
                            if data is None or len(data) == 0:
                                break

                            print(data[-1].__attributes__)

                            columns = ['date', 'text']
                             
                            df = pd.DataFrame(data, columns=data[-1].__attributes__)
                            
                            if 'timestamp' in df.columns:
                                df['date'] = pd.to_datetime(df['timestamp']).apply(lambda x:
                                                                                   x.date()
                                                                                   )
                                print('!')


                            if pd.unique(df['date'])[0] == None:

                                #attempt to reconstruct date from url, drop values that can't be converted 
                                df['date'] = pd.to_datetime(
                                                df['url'].apply(lambda x:'/'.join(
                                                    x.split('news/')[-1].split('/')[:3])
                                                                ),
                                                errors='coerce'
                                                ).dropna(how='any',
                                                axis=0
                                                ).reset_index(drop=True).apply(lambda x:x.date())
                            df = df[columns]
                        
                            yield df 
                            
                            lines_cnt += df.shape[0]
                            self.num_elements += df.shape[0]
                            progress.update(self.slice_size) 

                case _:
                    if verbose:
                        print(f'Unknown datatype to process: {data_path.split(".")}')
                    yield None

        except:
            if verbose:
                print(f'Load failed for file {data_path} with loading function {load_func}')
            yield None

    def load(self,
             verbose: bool = True
             ) -> None:

        '''
        Opens archives and saves their contents in a single file

        Inputs:
            self.data_dir:str | Path = NEWS_DATA_DIR - inputs in the format (function_to_process_file, file_names)
            self.slice_size: int = 10_000 - size of a chunk DataFrame to load from each archive
            self.delete_old: bool = True - flag to delete the old file
            self.unified_filepath: str = 'all_data.csv' - name of the new unified file
            verbose: bool = True - verbosity flag

        Outputs:
            None

        '''

        if os.path.exists(self.unified_filepath) and self.delete_old:
            if verbose:
                print('Found file with the same name in the data directory, deleting...')
            os.remove(self.unified_filepath)
        

        pd.DataFrame(columns=['date', 'text']).to_csv(self.unified_filepath,
                                                      header=True)
        if os.path.exists('checkpoint.pkl'):
            with open('checkpoint.pkl', 'rb') as file:
                checkpoint: dict[str, int] = pickle.load(file)
        else:
            checkpoint = {'load_func' : 0,
                          'data_path' : 0,
                          'df' : 0,
                          'dt' : 0
                          }

        def save_checkpoint() -> None:
            with open('checkpoint.pkl', 'wb') as file:
                pickle.dump(checkpoint, file)
            return None

        total_nans = 0

        for load_func, data_paths in tqdm(func_to_data.items(),
                                          desc='Loading data',
                                          unit=' file batches',
                                          colour='green',
                                          disable=not verbose):

            #print(data_paths)
            if self.unified_filenm in data_paths:
                data_paths.remove(self.unified_filenm)

            for idx, filenm in tqdm(enumerate(data_paths),
                                    desc='Adding paths',
                                    leave=False,
                                    disable=not verbose):

                data_paths[idx] = os.path.join(self.data_dir, filenm) #integrate full path with filename
           
            if len(data_paths):
                data_path = data_paths[0]
            else:
                continue


            for data_path in tqdm(data_paths,
                                  desc=f'Loading archive {data_path}',
                                  leave=False,
                                  unit=" files",
                                  disable=not verbose,
                                  initial=checkpoint['data_path']):

                for df in tqdm(self._prepare_data(data_path = data_path,
                                                  load_func=load_func,
                                                  verbose=verbose),
                               unit= " batches",
                               leave=False,
                               disable=not verbose#,
                               #initial=checkpoint['df']
                               ):

                    if df is None:
                        continue

                    print(df.head())

                    total_nans += len(df[df["date"].isna()])
                    
                    df = df.dropna(subset='date')
                    '''
                    print(f'{data_path}\n\n{df}')
                    ans = input('break?')

                    if ans == 'yes':
                        break
                    '''

                    df.to_csv(self.unified_filepath,
                             header=False,
                              mode='a',
                              columns=['date', 'text']
                              )
                else:
                    continue
                break

            else:
                continue
            break

        print(f'Total nans found: {total_nans}')

        return None

    def to_sentiment(self,
                     new_filepath: str | None = None,
                     mdl_cfg: dict | Any | None = None,
                     batch_size: int = 100,
                     verbose: bool = True
                     )->None:
        
        if new_filepath is None:
            new_filepath = Path(self.unified_filepath).with_name('all_data_sentiment.csv')
        
        self.sentiment_filepath = new_filepath

        mdl = Sentiment_Model(cfg=mdl_cfg)
        mdl.eval()

        reader = pd.read_csv(self.unified_filepath, chunksize=batch_size) 
        pd.DataFrame(columns=['date', 'sentiment']).to_csv(new_filepath, header=True)
        
        with torch.no_grad():
       
            for df in tqdm(reader,
                           desc='Converting text to sentiment',
                           total=int(np.ceil(self.num_elements / batch_size)),
                           leave=False,
                           disable=not verbose,
                           unit=' batches'):
                
                #(batch_size, 1)
                sentiment_scores = mdl.forward(
                                df['text'].fillna('nothing').tolist()
                                )

                df['sentiment'] = sentiment_scores.cpu().numpy()
                
                df.to_csv(new_filepath,
                          header=False,
                          mode='a',
                          columns=['date', 'sentiment']
                          )
            return None

    #TODO: make the case for low memory, or just rewrite to polars
    def agg(self,
            agg_func: str | Callable = 'mean',
            verbose: bool = False
            ) -> pd.DataFrame:
        ''' 
        Aggregates the data for the sentiment
        '''

        #here I reset and drop 'index' cause the index col may be 'date' or smth
        df = pd.read_csv(self.sentiment_filepath).reset_index(drop=False).drop(columns='index')

        #just in case there will be unnamed cols due to indices and stuff
        df.drop(columns=[col for col in df.columns if 'Unnamed' in col])

        agg_sentiment_path = Path(self.sentiment_filepath).with_name('agg_sentiment.csv')
        if verbose:
            print('Aggregating data...')
        df_new = df.groupby(by='date').mean()
        if verbose:
            print('Data aggregated, converting into .csv format...')

        df_new.to_csv(agg_sentiment_path)

        #reset the path to fetch data
        self.unified_filenm =  agg_sentiment_path
        self.num_elements = len(df_new)
        return df_new
        
class Time_Series_Dataset(Abstract_Fin_Dataset):

    def __init__(self,
                 data_dir: str | Path = os.path.join('..', 'data', 'stock_data'),
                 batch_size: int = 32,
                 lookback: int = 10,
                 horizon: int = 10,
                 unified_filenm: str = 'stock_data.csv',
                 load_from_file: bool = True,
                 delete_old: bool = True,
                 ) -> None:

        #self.get_init_args(locals()) 
        super().__init__(**self.get_init_args(locals()))
        
    def load(self,
             st: str = '2014-01-01',
             end: str = '2026-01-01'
             ) -> None:
       
        '''
        tickers= list(
                set(
                    trading_listing()['SECID'].to_list()
                    )
                )[:10]
        '''

        tickers = ['GAZP', 'YNDX', 'NVTK', 'SBER', 'VTBR', 'LKOH', 'GMKN', 'NLMK', 'MGNT', 'AFKS', 'AFLT', 'MTSS', 'HYDR', 'FEES', 'ALRS', 'PLZL', 'CHMF', 'MAGN', 'MOEX', 'TATN', 'SNGS']

        tickers = tickers[:2]

        df = history(list(tickers),
             st=st,
             end=end,
             max_retries=20,
             retry_pause=10,
             verbose=True)

        #print(df.shape)
        #print(df.columns)
        #print(df[('GAZP', 'TRADEDATE')][df[('GAZP', 'CLOSE')].isna()])
        #print(df.head())


        def func(sub_df):
            sub_df = sub_df.T.droplevel(axis=1, level=0)
            sub_df = sub_df[['TRADEDATE', 'OPEN', 'CLOSE', 'HIGH', 'LOW']]
            sub_df.dropna(axis=0, inplace=True)
            return sub_df
        
        cols_to_save = ['TRADEDATE', 'OPEN', 'CLOSE', 'HIGH', 'LOW']
        df = df.loc[:, pd.IndexSlice[:, cols_to_save]]
        df.columns = df.columns.map('_'.join)

        date_cols = [col  for col in df.columns if col.endswith('TRADEDATE')]

        df_new = pd.DataFrame({'TRADEDATE' : pd.bdate_range(start=st,
                                                            end=end)
                               }
                              )
        for col in date_cols:
            df[col] = pd.to_datetime(df[col])
            stock_cols = [column for column in df.columns
                          if column.startswith( col.split('_')[0] )
                          ]
            df_new = df_new.merge(df[stock_cols],
                                  how='left',
                                  left_on='TRADEDATE',
                                  right_on=col
                                  ).drop(columns=[col])

        df_new = df_new.dropna(axis=0, how='any').reset_index(drop=True)
        print(df_new.shape)
        print(df_new.head())
        self.num_elements = len(df)
        df_new.to_csv(self.unified_filenm)
        self.columns = df_new.columns.to_list()
        self.num_elements = len(range(len(df_new) - self.lookback - self.horizon + 1))
        self.shape: tuple = (self.num_elements, df_new.shape[1])
        return None

    def __getitem__(self, idx: int, dates: bool = False) -> (torch.Tensor, torch.Tensor):
        
        x = pd.read_csv(self.unified_filenm,
                        skiprows=min(0, idx-self.lookback),
                        nrows=self.lookback,
                        index_col=0
                        ).iloc[:, 0 if dates else 1:]

        y = pd.read_csv(self.unified_filenm,
                        skiprows=min(0, idx),
                        nrows=self.horizon,
                        index_col=0
                        ).iloc[:, 0 if dates else 1:]

        if dates:
            return x, y
        
        else:
            print(x)
            x, y = torch.Tensor(x.to_numpy().astype(float)),\
                    torch.Tensor(y.to_numpy().astype(float))

            return x, y

class Sentiment_Model(nn.Module, ABC):

    def __init__(self,
                 model_name: str | None = None,
                 num_labels: int | None = None,
                 batch_size: int = 512,
                 device: str = 'cpu',
                 score: bool = False,
                 output_hidden_states: bool = True,
                 cfg: dict | None = None) -> None:

        super().__init__()

    
        self.model_name = model_name
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.device = device
        self.score = score
        self.output_hidden_states = output_hidden_states
        if isinstance(cfg, dict):
            self.__dict__.update(cfg)

        self.mdl = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            output_hidden_states = self.output_hidden_states
        )
      
        self.mdl.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        return

    #TODO: add desc for torch.Tensor
    def forward(self,
                text_inputs: Iterable[str]
               ) -> torch.Tensor:
        
        tokenized_inputs = self.tokenizer(
                text_inputs, 
                return_tensors="pt", 
                truncation=True, 
                padding=True,
                max_length = 512
            ).to(self.device)

        outs = self.mdl.forward(**tokenized_inputs)

        if self.score:
            #print(outs.logits.shape)
            probs = F.softmax(outs.logits, dim=-1)
            probs = probs[:, 2] - probs[:, 0]
            return probs
        else:
            return outs

class Quantum_Encoder(nn.Module):

    def __init__(self,
                 encoder_type: str = 'Amplitude',
                 wires: Iterable | None = None,
                 device: str = 'default.qubit',
                 pad_val: complex = 0,
                 out: bool = False,
                 n_layers: int = 1,
                 cfg: dict | None = None
                 ) -> None:

        '''
        Wrapper class for encoding via pennylane functions

        Inputs:
            features: Sequence - classical feature values to encode
            encoder_type: str - type of encoding scheme, currently supported are 'Amplitude' for amplitude encoding, 'Phase' for phase encoding and 'QAOA' for the encdoing strategy inspired by the QAOA
            wires: Iterable | None - wires (qubits) to encode in the quantum circuit
            pad_val: int | float | complex - value to pad the classical vector with
        Outputs:
            None

        '''

        #TODO: add assert statements

        super().__init__()

        if isinstance(cfg, dict):
            self.__dict__.update(cfg)

        else:

            #initialize the simulator
            self.wires = wires
            self.n_layers = n_layers
            self.device = device
            
            self.encoder_type = encoder_type
            self.pad_val = pad_val
            self.out = out
            self.circuit: Callable = None
            self.weights_shape: tuple | None = None

        self.dev = qml.device(self.device, wires=self.wires) 

        match self.encoder_type:

            case 'Amplitude':

                self.weights_shape = tuple() #no weights here


            case 'Phase':
               
                self.weights_shape = (self.n_layers, len(self.wires))

            case 'QAOA':

                self.weights_shape: tuple[int] = qml.QAOAEmbedding.shape(n_layers=self.n_layers,
                                                                         n_wires=len(self.wires)
                                                                        )

            case _:
                raise NotImplementedError
 

    def _init_circuit(self,
                     weights: Annotated[torch.Tensor, ('n_layers', 'custom_val')] | str = None
                     ) -> Callable:
        ''' 
        
        Initialize the circuit for the embedding
        
        Inputs:
            None

        Outputs:
            circuit
        
        '''

        pad_features: Callable[[Iterable, Iterable, int | float], list] = lambda features, wires, pad_val: list(features) + [pad_val]*(len(wires) - len(features))
    

        def cond_decorator(condition:bool,
                       decorator: Callable[[Callable], Callable]
                       ) -> Callable[[Any], Any]:
            '''
            Made to call the qml.qnode decorator conditionally
            '''


            if condition:
                return decorator
            else:
                return lambda x: x 


        match self.encoder_type:

            case 'Amplitude':
                
                self.embed_size = 1 << len(bin(len(self.wires)).split('b')[-1])

                assert bin(self.embed_size).split('b')[-1].count('1') == 1, "self.embed_size is not a power of 2"

                #here weights are for compatibility, they don't serve any meaningful purpose
                @cond_decorator(self.out, qml.qnode(self.dev, interface='torch'))
                def circuit(features: Sequence,
                            weights: Annotated[torch.Tensor, ('n_layers', 'custom_val')] = None,
                            pad_val: complex = self.pad_val,
                            **kwargs
                            ) -> torch.Tensor:

                    if weights is None:
                        weights = nn.Parameter(torch.randn(self.weights_shape))

                    qml.AmplitudeEmbedding(features,
                                           wires=self.wires,
                                           pad_with=pad_val,
                                           **kwargs
                                           )
                    if self.out:
                        return qml.state()
                    else:
                        return None
            
            case 'Phase':
               
                @cond_decorator(self.out, qml.qnode(self.dev, interface='torch'))
                def circuit(features: Sequence,
                            weights: str = 'Z', #consider it a hyperparameter
                            ) -> torch.Tensor:

                    features = pad_features(features, self.wires, self.pad_val)
                    qml.AngleEmbedding(features, self.wires, rotation=weights)
                   
                    if self.out:
                        return qml.state()
                    else:
                        return None
            
                    

            case 'QAOA':
     
                @cond_decorator(self.out, qml.qnode(self.dev, interface='torch'))
                def circuit(features : Sequence,
                            weights: Annotated[torch.Tensor, ('n_layers', Literal['1', '3', '2*n_qubits'])] |\
                                        None = weights, 
                            **kwargs
                            ) -> torch.Tensor:

                    if weights is None:
                        weights = nn.Parameter(weights)

                    features = pad_features(features, self.wires, self.pad_val) 
                    qml.QAOAEmbedding(features, weights, self.wires, **kwargs)
                    
                    if self.out:
                        return qml.state()
                    else:
                        return None

            case _:

                raise NotImplementedError

        self.circuit = circuit

        return circuit

    def forward(self, features):
        if self.circuit:
            return self.circuit(features)
        else:
            raise ValueError("Circuit wasn't initialized")


class Quantum_Kernel(nn.Module):

    def __init__(self,
                 layer_type : str = 'StronglyEntanglingLayers',
                 wires: Iterable = range(5),
                 layer_config : dict[str : Any] = None,
                 encoder: nn.Module | None = None,
                 device: str = 'default.qubit',
                 n_layers: int = 1,
                 encoder_config: dict | None = None,
                 cfg: dict | None = None
                 ) -> None:

        super().__init__()
        
        if isinstance(cfg, dict):
            self.__dict__.update(cfg)
        else:
            self.encoder = encoder
            self.encoder_config = encoder_config
            self.layer_type = layer_type
            self.n_layers = n_layers
            self.wires = wires
            self.device = device

            self.layer_config = layer_config

        self.dev = qml.device(self.device, wires=self.wires)
        
        if self.encoder is not None:
            self.encoder = self.encoder(cfg=self.encoder_config)

        self.mapping = {'SimplifiedTwoDesign' : qml.SimplifiedTwoDesign,
                        'StronglyEntanglingLayers' : qml.StronglyEntanglingLayers}
        

        self.weights_shape = self.mapping[self.layer_type].shape(n_layers=self.n_layers,
                                                                 n_wires=len(self.wires)
                                                                )
    def _init_circuit(self,
                      kernel_weights: Annotated[torch.Tensor, ('n_layers', 'n_wires', 3)] |\
                                        Annotated[[
                                                Annotated[torch.Tensor,
                                                          'n_wires'],
                                                Annotated[torch.Tensor,
                                                          ('n_layers',
                                                          'n_wires - 1',
                                                          '2')]
                                                ],
                                            ('angles for the layer of Pauli-Y rotations',
                                            'weights for each layer')
                                            ],
                      encoder_weights: torch.Tensor | str | None = None
                      ) -> Callable:

        #TODO: update for SimplifiedTwoDesign compatibility
        layer_config = {'weights' : kernel_weights,
                        'wires' : self.encoder.wires if self.encoder else self.wires
                        }

        if self.encoder is not None:
            self.encoder_circuit = self.encoder._init_circuit(encoder_weights)

        @qml.qnode(self.dev, interface='torch')
        def circuit(features: Sequence) -> np.ndarray | torch.Tensor:
           
            self.encoder_circuit(features)

            #here the number of layers is within the shape of the weights tensor
            self.mapping[self.layer_type](**layer_config)
                        
            return qml.state() #TODO:change to be more flexible

        self.circuit = circuit

        return self.circuit

    def forward(self,
                features: Sequence
                ) -> torch.Tensor:
    
        if not hasattr(self, 'encoder_circuit'):
            print('Initializing without encoder...')

        return self.circuit(features)

class TS_JOPA(nn.Module):

    def __init__(self,
                 time_series_model: nn.Module,
                 time_series_config: dict,
                 quantum_model: nn.Module,
                 quantum_model_config: dict,
                 quantum_dim: int,
                 quantum_stride: int,
                 quantum_depth: int, #depth of the circuit in operations
                 dim_post_quantum: int,
                 max_quantum_register_size: int = 5, #max register size in qubits
                 batch_size: int = 32) -> None:

        super().__init__()

        self.batch_size = batch_size
        self.prediction_length = time_series_config.prediction_length
        
        if hasattr(time_series_config, 'hidden_dim'):
            self.hidden_dim = time_series_config['hidden_dim']
        else:
            self.hidden_dim = -1

        #Time series model initialization
        self.time_series_model = time_series_model(time_series_config) #init model with config
        self.time_series_proj_dim = time_series_config.d_model #define the projection dim ,TODO: change to generalize the integration point
        

        #Quantum model initialization
        self.quantum_model = quantum_model(cfg=quantum_model_config)
        self.q_encoder_weights = nn.Parameter(
                                    torch.randn(
                                        self.quantum_model.encoder.weights_shape
                                        )
                                    )
        print(self.quantum_model.weights_shape)
        self.q_kernel_weights = nn.Parameter(torch.randn(self.quantum_model.weights_shape))
        self.quantum_model._init_circuit(encoder_weights=self.q_encoder_weights,
                                         kernel_weights=self.q_kernel_weights)
        self.quantum_dim = quantum_dim
        self.max_quantum_register_size = max_quantum_register_size
        self.quantum_stride = quantum_stride
        self.quantum_depth = quantum_depth
        self.dim_post_quantum = dim_post_quantum
        

        #compare the number of qubits needed to fully encompass the embed dim vs the max allowed register size
        self.n_qubits = min((1 << quantum_dim.bit_length()).bit_length() - 1,
                            self.max_quantum_register_size)
       
        if self.n_qubits > self.max_quantum_register_size:
            self.num_registers = range((1 << self.n_qubits) - 1, quantum_dim, quantum_stride)
        else:
            self.num_registers = 1

        #self.quantum_model.mapping holds a mapping from the names of the layers to their pennylane classes
        num_params_quantum_layer = self.quantum_model.mapping[self.quantum_model.layer_type].\
                shape(n_wires=self.n_qubits,
                      n_layers=1
                      )

        #post-quantum algorithm projection
        self.post_q_proj = nn.Linear(1 << self.n_qubits,
                                     self.time_series_proj_dim
                                     )

        self.out_proj = nn.Linear(self.time_series_proj_dim,
                                  time_series_config.prediction_length * num_features
                                  )

    def forward(self,
                time_series_inputs: Annotated[torch.Tensor,
                                              ('batch_size',
                                               'n_time_steps',
                                               'time_series_dim'
                                               )
                                              ],
                sentiment: Annotated[torch.Tensor,
                                       ('batch_size',
                                        1
                                        )
                                       ]
                ) -> torch.Tensor:
        '''
        time_series_inputs: Annotated[torch.Tensor,
                                              ('batch_size',
                                               'n_time_steps',
                                               'time_series_dim'
                                               )
                                              ] - time series tensor
        sentiment: Annotated[torch.Tensor,
                                       ('batch_size',
                                        1
                                        )
                                       ] - sentiment values
        '''
        
        ts_out = self.time_series_model(time_series_inputs).hidden_states[self.hidden_dim] #(batch, n_patches, embed_size, d_model)
        ts_out_agg = ts_out.mean(dim=(1,2))
        #print(f'Time-series embeds: {ts_out_agg.shape}')

        combined = ts_out_agg * sentiment
        
        if ts_out_agg.shape[1] < self.quantum_dim:
            pad = torch.zeros(self.batch_size,
                              self.quantum_dim - combined.shape[1],
                              device=combined.device
                              )

            combined = F.pad(combined,
                             pad=(0, self.quantum_dim - combined.shape[1]),
                             mode='constant',
                             value=0)
            #combined = torch.cat([combined, pad], dim=1)
        
        q_outs = [] #quantum_registers' outputs
        for idr, register in enumerate(range(0,
                                             self.num_registers,
                                             self.quantum_stride)
                                       ):
            to_encode = combined[:,
                                 idr : min(combined.shape[1],
                                              idr+(1 << self.max_quantum_register_size)
                                                )
                                 ]
            magnitudes = to_encode.norm(p=2, dim=1, keepdim=True)

            to_encode = F.normalize(to_encode,
                                    p=2,
                                    dim=1
                                    )*2*np.pi #normalize to [0, 2pi] to encode
            
            q_outs.append(self.quantum_model.forward(to_encode)*magnitudes)

        q_outs = torch.cat(q_outs, dim=1).float()
        print(f'q_outs: {q_outs.shape}') #(batch, )
        #print(q_outs.dtype)

        post_q_outs = self.post_q_proj(q_outs)
        print(f'post_q_outs: {post_q_outs.shape}')
        out = self.out_proj(post_q_outs).view(-1, self.prediction_length, num_features) # (batch, 10, 8)
        print(f'out: {out.shape}')
        return out


if __name__ == '__main__':
    # ----------  Model parameters ----------
    batch_size = 4
    seq_len = 10               # time series sequence length
    num_features = 4*2 #channels per stock \times num_stocks
    prediction_length = 10
    d_model = 8                # hidden dimension for PatchTST

    # Time series config (PatchTST)
    time_series_config = PatchTSTConfig(
        context_length=seq_len,
        prediction_length=prediction_length,
        d_model=d_model,
        num_input_channels=num_features,
        patch_length=5,
        num_attention_heads=2,
        num_hidden_layers=1,
        output_hidden_states=True
    )

    # Quantum model parameters
    quantum_dim = 15
    quantum_stride = 2
    quantum_depth = 1
    dim_post_quantum = d_model
    max_quantum_register_size = 5

    quantum_model_config = {
        'layer_type' : 'StronglyEntanglingLayers',
        'n_layers' : 2,
        'wires': range(4),
        'layer_config': None,
        'encoder': Quantum_Encoder,
        'device': 'default.qubit',
        'uncorr_wires': (),
        'encoder_config' : {'embed_size' : quantum_dim,
                            'encoder_type' : 'Amplitude',
                            'wires' : range(4),
                            'device' : 'default.qubit',
                            'pad_val' : 0,
                            'n_layers' : 2,
                            'out' : True
                            }
        }

    

    # ---------- Instantiate the combined model ----------
    combined_model = TS_JOPA(
        time_series_model=PatchTSTForPrediction,
        time_series_config=time_series_config,
        quantum_model=Quantum_Kernel,
        quantum_model_config=quantum_model_config,
        quantum_dim=quantum_dim,
        quantum_stride=quantum_stride,
        quantum_depth=quantum_depth,
        dim_post_quantum=dim_post_quantum,
        max_quantum_register_size=max_quantum_register_size
    )

    sentiment_cfg = {
            "model_name": "ProsusAI/finbert",
            "num_labels": 3,                  
            "device": "cuda" if torch.cuda.is_available() else 'cpu',
            "score" : True
            }

    #sentiment_model = Sentiment_Model(cfg=sentiment_cfg)

    #JOPA = TS_JOPA(combined_model, sentiment_model)

    #print("Combined model instantiated successfully.")

    # ---------- Prepare data ----------

    
    #path_news_data = os.path.join('..', 'data', 'news_data')
    #news_data = News_Dataset(path_news_data,
    #                         unified_filenm='finished_data_sentiment.csv',
    #                         load_from_file=True,
    #                         delete_old=False)

    #news_data.to_sentiment(batch_size=100, mdl_cfg=sentiment_cfg)
    #news_data.agg(verbose= True)
    #news_dataloader = DataLoader(news_data, batch_size=32)
    
    #time_series_inputs = Time_Series_Dataset(lookback=seq_len,
    #                                         horizon=prediction_length,
    #                                         load_from_file=False,
    #                                         delete_old=True)

    #time_series_batch_size = 32
    #time_series_dataloader = DataLoader(time_series_inputs,
    #                                    batch_size=time_series_batch_size,
    #                                    shuffle=False)


    batch_size = 32
    data = DataLoader(Joint_Dataset(lookback=seq_len,
                                    horizon=prediction_length),
                      batch_size=32,
                      shuffle=True
                      )

    num_epochs = 10


    #TODO: finish the data pipeline to have a unified dataset to load from. Preprocess news into sentiment values for each date
    for epoch in range(num_epochs):

        #TODO: fix later
        for sentiment, inputs, targets in data:


            output = combined_model(inputs, sentiment)
            print(f"Forward output shape: {output.shape}")   # (batch, prediction_length)
            print(f'Target shape: {targets.shape}')
            loss = nn.MSELoss()(output, targets)
            loss.backward()
            print("Backward pass completed. Gradients exist:", any(
                p.grad is not None for p in combined_model.parameters()
            ))

            break
        
        else:
            continue
        
        break
