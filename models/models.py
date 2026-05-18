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

from itertools import combinations, islice
from corus import load_lenta, load_lenta2, load_mokoron, load_buriy_news, load_buriy_webhose
from torch.utils.data import DataLoader, ConcatDataset, Dataset, Subset, TensorDataset

from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification, 
        PatchTSTConfig, PatchTSTForPrediction
        )

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression

from tqdm import tqdm

from moex_api.history import history, trading_listing

from pathlib import Path
from typing import Callable, Generator, Any, Iterable, Sequence, Annotated, Literal, Optional
from abc import abstractmethod, ABC

seed = 42

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)      # if using multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
                 lookback: int = None,
                 horizon: int = None
                 ) -> None:

        super().__init__()

        if lookback is None:
            lookback = 0
        if horizon is None:
            horizon = 0

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_elements = 0
        self.unified_filenm = unified_filenm
        self.unified_filepath = os.path.join(data_dir, unified_filenm)
        self.delete_old = delete_old
        
        if not load_from_file:
            self.load()
        else:
            self.df = pd.read_csv(self.unified_filepath)
            self.num_elements = sum(len(chunk) for chunk in pd.read_csv(self.unified_filepath, chunksize=10_000)) - lookback - horizon + 1
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
        df_new = df.groupby(by='date').mean().reset_index()
        if verbose:
            print('Data aggregated, converting into .csv format...')

        df_new.to_csv(agg_sentiment_path, index=False)

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
             st: str = '2004-01-01',
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

    def transform(self,
                  pipe,
                  size: float = .8
                  ) -> None:
       
        redundant_cols = ['sentiment', 'TRADEDATE']
        redundant = self.df[redundant_cols]

        cutoff = int(size*len(self.df))
        train, test = self.df.iloc[:cutoff, :].drop(columns=['sentiment', 'TRADEDATE']),\
                    self.df.iloc[cutoff:, :].drop(columns=['sentiment', 'TRADEDATE'])

        cols = train.columns
        train = pd.DataFrame(data=pipe.fit_transform(train))
        test = pd.DataFrame(data=pipe.transform(test))

        #print(train.head(), test.head())

        self.df = pd.concat([train, test], axis=0).reset_index(drop=True)
        self.df = pd.concat([self.df, redundant], axis=1)

        return None

    def __getitem__(self,
                    idx: int
                    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        
        real_idx = self.lookback + idx

        #print(self.df)

        #reshaped to (batch, 1)
        sentiment = torch.Tensor(np.array([self.df.loc[real_idx, 'sentiment']]).\
                                                                            reshape(-1, 1).\
                                                                            astype(float)
                                 )
    
        x = self.df.iloc[real_idx - self.lookback: real_idx, :].drop(columns=['TRADEDATE',
                                                                                'sentiment']
                                                                                                              ).to_numpy().astype(float)

        #print(x)
        #print(x.shape)

        y = self.df.iloc[real_idx: real_idx + self.horizon, :].drop(columns=['TRADEDATE',
                                                                              'sentiment']
                                                                     ).to_numpy().astype(float)

        
        x, y = torch.Tensor(x),torch.Tensor(y)

        return sentiment, x, y

class PatchTST(nn.Module):
    
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.model = PatchTSTForPrediction(config)
        self.prediction_length = config.prediction_length
        self.num_features = config.num_input_channels

    def forward(self,
                time_series_inputs: torch.Tensor,
                sentiment: torch.Tensor = None
                ) -> torch.Tensor:
        '''
        time_series_inputs: (batch_size, seq_len, num_features)
        sentiment: ignored, kept for API compatibility
        Returns: (batch_size, prediction_length, num_features)
        '''
        outputs = self.model(time_series_inputs)
        return outputs.prediction_outputs


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
                 device: str = None,
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

        if device is None:
            device = 'lightning.gpu' if torch.cuda.is_available() else 'lightning.qubit'

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
        self.weights = None

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

    def _init_circuit(self, weights=None):
        if weights is not None:
            self.weights = nn.Parameter(weights)
        else:
            self.weights = None

        @qml.batch_params
        @qml.qnode(self.dev, interface='torch')
        def circuit(features, weights):
            feat = features
            
            if len(feat) > len(self.wires):
                feat = feat[:len(self.wires)]
            elif len(feat) < len(self.wires):
                feat += [0.0] * (len(self.wires) - len(feat))

            if self.encoder_type == 'Amplitude':
                qml.AmplitudeEmbedding(feat, wires=self.wires, pad_with=self.pad_val)
            elif self.encoder_type == 'Phase':
                qml.AngleEmbedding(feat, wires=self.wires, rotation='Z')
            elif self.encoder_type == 'QAOA':
                qml.QAOAEmbedding(feat, weights, self.wires)
            else:
                raise NotImplementedError

            if self.out:
                return qml.state()
            else:
                return None

        self.circuit = circuit
        return circuit

    def forward(self, features):
        if self.circuit:
            if features.dim() == 2:
                batch_size = features.size(0)
                outputs = []
                for i in range(batch_size):
                    out = self.circuit(features[i], self.weights)   
                    outputs.append(out)
                return torch.stack(outputs, dim=0)
            else:
                return self.circuit(features, self.weights)
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
    def _init_circuit(self, kernel_weights=None, encoder_weights=None):
        if kernel_weights is not None:
            self.kernel_weights = nn.Parameter(kernel_weights)
        
        if encoder_weights is not None and self.encoder is not None:
            self.encoder_weights = nn.Parameter(encoder_weights)
            self.encoder._init_circuit(self.encoder_weights)

        @qml.batch_params
        @qml.qnode(self.dev, interface='torch')
        def circuit(features, encoder_weights, kernel_weights):
            if self.encoder is not None:
                if encoder_weights is not None:
                    self.encoder.circuit(features, encoder_weights)
                else:
                    self.encoder.circuit(features)
            if kernel_weights is not None:
                self.mapping[self.layer_type](kernel_weights, wires=self.wires)
            else:
                self.mapping[self.layer_type](wires=self.wires)
            res = [qml.expval(qml.PauliZ(w)) for w in self.wires]
            return res

        self.circuit = circuit
        return circuit

    def forward(self,
                features: Sequence,
                sentiment
                ) -> torch.Tensor:
    
        if features.dim() == 2:
            batch_size = features.size(0)
            outputs = []
            for i in range(batch_size):
                out = self.circuit(features[i],
                                   self.encoder_weights,
                                   self.kernel_weights
                                   )
                out = torch.stack(out)
                outputs.append(out)
            return torch.stack(outputs, dim=0)
        else:
            return self.circuit(features,
                                self.encoder_weights,
                                self.kernel_weights
                                )



class QuantumCircuit(nn.Module):
    '''
    Unified quantum module that performs both encoding and variational processing.
    '''
    
    def __init__(self,
                 n_qubits: int,
                 n_steps: int,
                 horizon: int,
                 batch_size: int,
                 encoding_type: str = 'Phase',
                 n_encoding_layers: int = 1,
                 var_layer_type: str = 'StronglyEntanglingLayers',
                 n_var_layers: int = 2,
                 device: str = None,
                 out: bool = False, 
                 pad_val: complex = 0.0,
                 apply_qft: bool = True,
                 cfg: Optional[dict] = None):
        """
        Args:
            n_qubits: number of qubits
            encoding_type: 'Amplitude', 'Phase', or 'QAOA'
            n_encoding_layers: number of layers for QAOA encoding (ignored for others)
            var_layer_type: variational layer type
            n_var_layers: number of variational layers
            device: PennyLane device ('default.qubit', 'lightning.qubit', etc.)
            out: if True, return full state vector, else expectation values
            pad_val: padding value for amplitude encoding
            cfg: optional dict to override attributes
        """
        
        super().__init__()
        
        if cfg is not None:
            self.__dict__.update(cfg)
            return
        
        self.n_qubits = n_qubits
        self.n_steps = n_steps
        self.horizon = horizon
        self.batch_size = batch_size
        self.wires = range(n_qubits)
        self.encoding_type = encoding_type
        self.n_encoding_layers = n_encoding_layers
        self.var_layer_type = var_layer_type
        self.n_var_layers = n_var_layers
        self.out = out
        self.pad_val = pad_val
        self.apply_qft = apply_qft
        
        if device is None:
            device = 'lightning.gpu' if torch.cuda.is_available() else 'lightning.qubit'
        self.device = device
        self.dev = qml.device(device, wires=self.wires)
        
        # Determine weight shapes
        self.encoder_weights_shape = None
        
        match self.encoding_type:

            case 'Amplitude':

                self.encoder_weights_shape = (self.batch_size, ) + tuple() #no weights here

            case 'Phase':
               
                self.encoder_weights_shape = (self.batch_size, ) + (self.n_encoding_layers, len(self.wires))
                
            case 'QAOA':

                self.encoder_weights_shape: tuple[int] = (self.batch_size, ) +\
                                        qml.QAOAEmbedding.shape(n_layers=self.n_encoding_layers,
                                                                n_wires=len(self.wires)
                                                                )
            case _:
                raise NotImplementedError

        var_layer_map = {
            'StronglyEntanglingLayers': qml.StronglyEntanglingLayers,
            'SimplifiedTwoDesign': qml.SimplifiedTwoDesign
        }
        if var_layer_type not in var_layer_map:
            raise ValueError(f"Unsupported variational layer: {var_layer_type}")
        
        self.var_layer_fn = var_layer_map[var_layer_type]
        
        self.var_weights_shape = (self.batch_size, ) + self.var_layer_fn.shape(
                                                                n_layers=n_var_layers,
                                                                n_wires=n_qubits
                                                                )
        
        self.circuit = None
        self.encoder_weights = None
        self.var_weights = None


        if self.encoding_type == 'QAOA' and self.encoder_weights_shape:
            self.encoder_weights = torch.randn(*self.encoder_weights_shape) * torch.pi

        if self.var_weights_shape:
            self.var_weights = torch.randn(*self.var_weights_shape) * torch.pi

        self._init_circuit(self.encoder_weights, self.var_weights)
    
        self.proj = nn.Linear(self.batch_size, self.horizon)

    def _init_circuit(self, encoder_weights=None, var_weights=None):
        
        if encoder_weights is not None:
            self.encoder_weights = nn.Parameter(encoder_weights)
        
        if var_weights is not None:
            self.var_weights = nn.Parameter(var_weights)
        
        @qml.batch_params
        @qml.qnode(self.dev, interface='torch')
        def circuit(features, enc_weights, var_weights):
            
            for step in range(self.n_steps):
                step_feat = features[:, step, :]           # (batch_size, n_features)
                step_feat = self._pad_features(step_feat)  # (batch_size, n_qubits)

                if self.encoding_type == 'Amplitude':
                    qml.AmplitudeEmbedding(step_feat, wires=self.wires,
                                           pad_with=self.pad_val, normalize=True)
                elif self.encoding_type == 'Phase':
                    qml.AngleEmbedding(step_feat, wires=self.wires, rotation='Z')
                
                elif self.encoding_type == 'QAOA':
                    qml.QAOAEmbedding(step_feat, enc_weights, wires=self.wires)
                
                else:
                    raise NotImplementedError

                if var_weights is not None:
                    self.var_layer_fn(var_weights, wires=self.wires)
                else:
                    self.var_layer_fn(wires=self.wires)

            if self.apply_qft:
                qml.QFT(wires=self.wires)

            return [qml.expval(qml.PauliZ(w)) for w in self.wires]    
        
        self.circuit = circuit

    def _pad_features(self, x: torch.Tensor) -> torch.Tensor:
        n_features = x.shape[-1]
        if n_features > self.n_qubits:
            return x[..., :self.n_qubits]
        elif n_features < self.n_qubits:
            pad = torch.zeros(*x.shape[:-1], self.n_qubits - n_features,
                              dtype=x.dtype, device=x.device)
            return torch.cat([x, pad], dim=-1)
        return x


    def forward(self, features: torch.Tensor, sentiment) -> torch.Tensor:
        '''
        Forward pass with batching support.
        Args:
            features: shape (batch_size, n_features) – features to encode.
                      n_features can be different from n_qubits (will be padded/truncated).
        Returns:
            If out=False: shape (batch_size, n_qubits) – expectation values.
            If out=True: shape (batch_size, 2**n_qubits) – state vector (complex).
        '''
        if self.circuit is None:
            raise RuntimeError("Circuit not initialized. Call _init_circuit first.")
        
        expvals = self.circuit(features, self.encoder_weights, self.var_weights)
        q_out = torch.stack(expvals, dim=1).float()

        out = self.proj(q_out)
        out = out.reshape(self.batch_size, self.horizon, self.n_qubits)
        return out


class TS_JOPA(nn.Module):

    def __init__(self,
                 time_series_model: nn.Module,
                 time_series_config: dict,
                 quantum_model: nn.Module,
                 quantum_model_config: dict,
                 n_qubits: int,                  
                 dim_post_quantum: int,
                 batch_size: int = 32) -> None:

        super().__init__()
        self.batch_size = batch_size
        self.prediction_length = time_series_config.prediction_length
        self.num_features = time_series_config.num_input_channels
        self.n_qubits = n_qubits
        self.hidden_dim = getattr(time_series_config, 'hidden_dim', -1)

        self.time_series_model = time_series_model(time_series_config)
        self.time_series_proj_dim = time_series_config.d_model

        self.quantum_model = quantum_model(cfg=quantum_model_config)
        encoder_weights = None
        if self.quantum_model.encoder is not None:
            if hasattr(self.quantum_model.encoder, 'weights_shape') and self.quantum_model.encoder.weights_shape:
                encoder_weights = torch.pi * torch.randn(*self.quantum_model.encoder.weights_shape)
        kernel_weights = None
        if hasattr(self.quantum_model, 'weights_shape') and self.quantum_model.weights_shape:
            kernel_weights = torch.pi * torch.randn(*self.quantum_model.weights_shape)
        self.quantum_model._init_circuit(kernel_weights=kernel_weights, encoder_weights=encoder_weights)

        self.post_q_proj = nn.Linear(n_qubits, self.time_series_proj_dim)
        self.out_proj = nn.Linear(self.time_series_proj_dim,
                                  time_series_config.prediction_length * self.num_features)        


    def forward(self, time_series_inputs, sentiment):
        sentiment = sentiment.reshape(-1, 1)
        ts_out = self.time_series_model(time_series_inputs).hidden_states[self.hidden_dim]
        ts_out_agg = ts_out.mean(dim=(1, 2))
        combined = ts_out_agg * sentiment

        if combined.shape[1] < self.n_qubits:
            combined_q = F.pad(combined, (0, self.n_qubits - combined.shape[1]), value=0.0)
        else:
            combined_q = combined[:, :self.n_qubits]

        norms = combined.norm(dim=1, keepdim=True) + 1e-8
        combined = combined / norms * (2 * torch.pi)

        q_out = self.quantum_model(combined).real.float() * norms / (2* torch.pi)# (batch, n_qubits)
        post_q = self.post_q_proj(q_out)               # (batch, d_model)
        out = self.out_proj(post_q).view(-1, self.prediction_length, self.num_features)
        return out


def train(dataset: torch.utils.data.Dataset,
          mdl: nn.Module,
          batch_size: int = 32,
          criterion = nn.MSELoss(),
          num_epochs: int = 10,
          train_pct: float = .8
          ) -> None:

    cutoff = int(train_pct*len(dataset))

    train_indices = list(range(cutoff))
    test_indices = list(range(cutoff, len(dataset)))

    train_dataset = Subset(dataset, train_indices)
    test_dataset  = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(mdl.parameters(),
                                  lr=1e-4,
                                  weight_decay=1e-5
                                 )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           patience=3,
                                                           factor=0.5
                                                           )
    best_test_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 10

    for epoch in range(num_epochs):

        mdl.train()
        train_loss = 0.0
        train_loop = tqdm(train_loader,
                          desc=f'Epoch {epoch+1}/{num_epochs} [Train]',
                          leave=False
                          )


        for sentiment, inputs, targets in train_loader:

            sentiment, inputs, targets = sentiment.to(device),\
                                            inputs.to(device),\
                                            targets.to(device)



            optimizer.zero_grad()
            outputs = mdl(inputs, sentiment)
            loss = criterion(outputs, targets)
            loss.backward()

            #gradient clipping 
            #torch.nn.utils.clip_grad_norm_(mdl.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_loop.set_postfix(loss=loss.item())

            #break

        else:
            avg_train_loss = train_loss / len(train_loader.dataset)
        
            #check on test
            mdl.eval()
            test_loss = 0.0
            with torch.no_grad():
                test_loop = tqdm(test_loader,
                                 desc=f'Epoch {epoch+1}/{num_epochs} [Test]',
                                 leave=False)

                for sentiment, inputs, targets in test_loop:
                    sentiment, inputs, targets = sentiment.to(device), inputs.to(device), targets.to(device)
                    outputs = mdl(inputs, sentiment)
                    loss = criterion(outputs, targets)
                    test_loss += loss.item() * inputs.size(0)
                    test_loop.set_postfix(loss=loss.item())
            
            avg_test_loss = test_loss / len(test_loader.dataset)
            scheduler.step(avg_test_loss)
            
            print(f'\rEpoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Test Loss = {avg_test_loss:.6f}')
            

            if (epoch + 1) % 5 == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': mdl.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_test_loss,
                }
                torch.save(checkpoint, f"checkpoint_epoch_{epoch+1}.pth")

    
            # early stopping
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                torch.save(mdl.state_dict(), 'best_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print("\nEarly stopping triggered.")
                    break


    print("Training complete. Best test loss:", best_test_loss)
    return None


class LogReturnsTransformer(BaseEstimator, TransformerMixin):
    """Compute log returns for OHLC columns."""
    def __init__(self, cols: Iterable = ['OPEN', 'HIGH', 'LOW', 'CLOSE']):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        nms = set([col.split('_')[0] for col in X.columns])
        for nm in nms:
            for col in self.cols:
                X[f'{nm}_{col}'] = np.log(X[f'{nm}_{col}'] / X[f'{nm}_{col}'].shift(1))
        # Drop the first row (NaN from shift)
        X = X.fillna(0)#X.iloc[1:].reset_index(drop=True)
        return X


class SpreadTransformer(BaseEstimator, TransformerMixin):
    """Add ratio features: High/Low, Close/Open."""
    def transform(self, X):
        X = X.copy()
        nms = set([col.split('_')[0] for col in X.columns])
        for nm in nms:
            X[f'{nm}_HIGH_LOW_ratio'] = X[f'{nm}_HIGH'] / X[f'{nm}_LOW']
            X[f'{nm}_CLOSE_OPEN_ratio'] = X[f'{nm}_CLOSE'] / X[f'{nm}_OPEN']
        return X

    def fit(self, X, y=None):
        return self


class TypicalPriceTransformer(BaseEstimator, TransformerMixin):
    """Add typical price = (High + Low + Close)/3."""
    def transform(self, X):
        X = X.copy()
        nms = set([col.split('_')[0] for col in X.columns])
        for nm in nms:
            X[f'{nm}_Typical'] = (X[f'{nm}_HIGH'] + X[f'{nm}_LOW'] + X[f'{nm}_CLOSE']) / 3.0
        return X

    def fit(self, X, y=None):
        return self


class RollingRSITransformer(BaseEstimator, TransformerMixin):
    """Add RSI (Relative Strength Index) with a given window."""
    def __init__(self, window=14, col='Close'):
        self.window = window
        self.col = col

    def transform(self, X):
        X = X.copy()
        delta = X[self.col].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=self.window, min_periods=self.window).mean()
        avg_loss = loss.rolling(window=self.window, min_periods=self.window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        X[f'RSI_{self.window}'] = rsi
        # First 'window' rows will be NaN – we will drop them later
        return X

    def fit(self, X, y=None):
        return self


class Quantum_feature_map(BaseEstimator):
    def __init__(self,
                 MI_threshold=2.5,
                 n_qubits=4
                 ) -> None:

        self.MI_th = MI_threshold
        self.n_qubits = n_qubits
        self.device = 'default.qubit'

    def fit(self, X, y=None):
        X = np.array(X)
        n_features = X.shape[1]

        MI = np.zeros((n_features, n_features))
        for i in range(n_features):
            MI[:, i] = mutual_info_regression(X, X[:, i])

        #print(MI)

        groups = []
        for i, j in combinations(range(n_features), 2):
            if (MI[i, j] >= self.MI_th or MI[j, i] >= self.MI_th) and i != j:
                groups.append((i, j))

        self.groups_ = groups
        self.params_ = [MI[gr] for gr in groups]

        dev = qml.device(self.device, wires=self.n_qubits)

        @qml.qnode(dev)
        def circuit(inputs):
            for i in range(min(len(inputs), self.n_qubits)):
                qml.RX(inputs[i], wires=i)
            
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            
            for group, angle in zip(self.groups_, self.params_):
                qml.CRZ(angle, wires=group)

            qml.QFT(wires=range(self.n_qubits))
            
            return [qml.expval(qml.PauliZ(w)) for w in range(self.n_qubits)] 

        self.circuit_ = circuit
        return self

    def transform(self, X):
        X = np.array(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        out = np.zeros((X.shape[0], self.n_qubits))
        for idx in range(X.shape[0]):
            # Store the quantum output
            out[idx] = self.circuit_(X_scaled[idx])

        # Inverse transform to original scale (optional)
        X_new = scaler.inverse_transform(out)
        return X_new

if __name__ == '__main__':

    n_stocks = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_pct = .8
    num_epochs = 10
    batch_size = 32
    seq_len = 10               # time series sequence length
    num_features = 4*n_stocks  #4*5 #channels per stock \times num_stocks
    prediction_length = 10
    d_model = 8                # hidden dimension for PatchTST
    num_hidden_layers = 2
    dropout= .2              
    head_dropout= .2
    

    # Time series config (PatchTST)
    time_series_config = PatchTSTConfig(
        context_length=seq_len,
        prediction_length=prediction_length,
        d_model=d_model,
        num_input_channels=num_features,
        patch_length=5,
        num_attention_heads=2,
        num_hidden_layers=num_hidden_layers,
        dropout=dropout,
        head_dropout=head_dropout,
        output_hidden_states=True
    )

    # Quantum model parameters
    n_qubits = 4
    dim_post_quantum = d_model

    quantum_model_config = {
        'layer_type': 'StronglyEntanglingLayers',
        'n_layers': 4,
        'wires': range(n_qubits),
        'layer_config': None,
        'encoder': Quantum_Encoder,
        'device': 'default.qubit',
        'uncorr_wires': (),
        'encoder_config': {
            'encoder_type': 'Phase',
            'wires': range(n_qubits),
            'device': 'default.qubit',
            'n_layers': 2,
            'out': True
        }
    }

    combined_model = TS_JOPA(
        time_series_model=PatchTSTForPrediction,
        time_series_config=time_series_config,
        quantum_model=Quantum_Kernel,
        quantum_model_config=quantum_model_config,
        n_qubits=n_qubits,                    
        dim_post_quantum=dim_post_quantum,
        batch_size=batch_size
    ).to(device)

    full_dataset = Joint_Dataset(lookback=seq_len,
                                 horizon=prediction_length,
                                 load_from_file=True,
                                 unified_filenm=f'preprocessed_data_{n_stocks}_stocks.csv') 

    pipe = Pipeline([
        #('QMI', Quantum_feature_map(n_qubits=num_features, MI_threshold=2.)),
        #('log_returns', LogReturnsTransformer()),
        #('spreads', SpreadTransformer()),
        #('typical', TypicalPriceTransformer()),
        #('rsi', RollingRSITransformer(window=14, col='CLOSE')),
        #('scaler', StandardScaler()),
        #('PCA', PCA(n_components=num_features)),
        ('scaler_2', StandardScaler()),
    ])


    full_dataset.transform(pipe, train_pct)

    benchmark_model = PatchTST(time_series_config)

    quantum_model = QuantumCircuit(
        n_qubits=n_qubits,
        n_steps=seq_len,
        horizon=prediction_length,
        batch_size=batch_size,
        encoding_type='QAOA',
        var_layer_type='StronglyEntanglingLayers',
        n_var_layers=2,
        device='default.qubit'
    )
    
    encoder_weights = None
    var_weights = torch.randn(*quantum_model.var_weights_shape) * torch.pi
    quantum_model._init_circuit(encoder_weights=encoder_weights, var_weights=var_weights)    

    criterion=nn.MSELoss()

    train(full_dataset,
          quantum_model,
          batch_size,
          criterion=criterion,
          num_epochs=num_epochs,
          train_pct=train_pct
          ) 

    '''


    train(full_dataset,
          benchmark_model,
          batch_size,
          criterion=criterion,
          num_epochs=num_epochs,
          train_pct=train_pct
          )
    '''


    '''
    train(full_dataset,
          combined_model,
          batch_size,
          criterion=criterion,
          num_epochs=num_epochs,
          train_pct=train_pct)
    '''
