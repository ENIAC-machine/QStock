import os
import sys
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
from typing import Callable, Generator, Any, Iterable, Sequence, Annotated
from abc import abstractmethod, ABC

DATA_DIR = os.path.join('..', 'data')

#global variable to map preprocessing functions to the files
func_to_data = {load_lenta: ['lenta-ru-news.csv.gz'],
                load_lenta2 : ['lenta-ru-news.csv.bz2'],
                load_mokoron : ['db.sql'],
                load_buriy_news : ['news-articles-2014.tar.bz2',
                                   'news-articles-2015-part1.tar.bz2',
                                   'news-articles-2015-part2.tar.bz2'],
                load_buriy_webhose : ['webhose-2016.tar.bz2'],
                pd.read_csv : [file for file in os.listdir(DATA_DIR) if file.endswith('.csv')]
                }


class Abstract_Fin_Dataset(Dataset, ABC):

    def __init__(self,
                 data_dir: str | Path = os.path.join('..', 'data'),
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
        self.delete_old = delete_old
        
        if not load_from_file:
            self.load()

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

    def __len__(self) -> int:
        return self.num_elements

class News_Dataset(Abstract_Fin_Dataset):

    def __init__(self,
                 data_dir: str | Path = os.path.join('.'),
                 batch_size: int = 32,
                 slice_size: int = 10_000,
                 unified_filenm: str = 'all_data.csv',
                 load_from_file: bool = False,
                 delete_old: bool = True,
                 **kwargs
                 ) -> None:

        self.slice_size = slice_size
        
        super().__init__()

    def _prepare_data(self,
                      data_path: str | Path,
                      load_func: Callable,
                      verbose: bool = True
                      ) -> Generator[pd.DataFrame | None]:

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

                        columns = [col for col in data[-1].__attributes__
                                            if col in ['date', 'text', 'timestamp']
                                   ]
                    
                        df = pd.DataFrame(data, columns=data[-1].__attributes__)[columns]
                        yield df 
                        lines_cnt += df.shape[0]
                        self.num_elements += df.shape[0]
                        progress.update(self.slice_size) 

            case _:
                print(f'Unknown datatype to process: {data_path.split(".")}')
                yield None


    def load(self,
             verbose: bool = True
             ) -> None:

        '''
        Opens archives and saves their contents in a single file

        Inputs:
            self.data_dir:str | Path = DATA_DIR - inputs in the format (function_to_process_file, file_names)
            self.slice_size: int = 10_000 - size of a chunk DataFrame to load from each archive
            self.delete_old: bool = True - flag to delete the old file
            self.unified_filenm: str = 'all_data.csv' - name of the new unified file
            verbose: bool = True - verbosity flag

        Outputs:
            None

        '''

        path_save = os.path.join(self.data_dir, self.unified_filenm)

        if os.path.exists(path_save) and self.delete_old:
            if verbose:
                print('Found file with the same name in the data directory, deleting...')
            os.remove(path_save)
        

        header = True
        df_tmp = pd.DataFrame(columns=['date', 'text']).to_csv(path_save, header=header)
        header = False

        for load_func, data_paths in tqdm(func_to_data.items(),
                                          desc='Loading data',
                                          unit=' file batches',
                                          colour='green',
                                          disable=not verbose):

            for idx, filenm in tqdm(enumerate(data_paths),
                                    desc='Adding paths',
                                    leave=False,
                                    disable=not verbose):
                data_paths[idx] = os.path.join(self.data_dir, filenm) #integrate full path with filename
            
            data_path = data_paths[0]

            for data_path in tqdm(data_paths,
                                  desc=f'Loading archive {data_path}',
                                  leave=False,
                                  unit=" files",
                                  disable=not verbose):

                for df in self._prepare_data(data_path, load_func, verbose):

                    if 'date' not in df.columns:
                        df['date'] = pd.to_datetime(df['timestamp']).apply(lambda x: x.date())
                        
                    df.to_csv(path_save,
                              header=header,
                              mode='a',
                              columns=['date', 'text']
                              )

        return None 


    def __getitem__(self, idx: int | Iterable[int]) -> list[Any]:
        return pd.read_csv(self.unified_filenm,
                           skip_rows=idx[0] if isinstance(idx, Iterable) else idx,
                           nrows=len(idx) if isinstance(idx, Iterable) else 1
                           ).iloc[[idx], :].to_list()


class Time_Series_Dataset(Abstract_Fin_Dataset):

    def __init__(self,
                 data_dir: str | Path = os.path.join('..', 'data', 'stock_data'),
                 batch_size: int = 32,
                 unified_filenm: str = 'stock_data.csv',
                 load_from_file: bool = True,
                 delete_old: bool = True,
                 ) -> None:

        super().__init__()
        
        self.current_stock = pd.read_csv(self.unified_filenm, nrows=1)['TICKER'].value

    def load(self) -> None:
        tickers= list(set(trading_listing(status='traded')['SECID'].to_list()))

        df = history(list(tickers),
             st='2000-01-01',
             end='2026-01-01',
             max_retries=20,
             retry_pause=4,
             verbose=True)

        def func(sub_df):
            sub_df = sub_df.T.droplevel(axis=1, level=0)
            sub_df = sub_df[['TRADEDATE', 'OPEN', 'CLOSE', 'HIGH', 'LOW']]
            sub_df.dropna(axis=0, inplace=True)
            return sub_df
        
        df = df.T.groupby(level=0).apply(func).reset_index().\
                rename(columns={'DataFrame' : 'TICKER'}).\
                set_index('TRADEDATE').drop(columns='level_1').\
                sort_values('TRADEDATE')
                
        self.num_elements = len(df)
        df.to_csv(self.unified_filenm)
        return None

    def __getitem__(self, idx: int | Iterable[int]) -> list[Any]:
        return pd.read_csv(self.unified_filenm,
                           skip_rows=idx[0] if isinstance(idx, Iterable) else idx,
                           nrows=len(idx) if isinstance(idx, Iterable) else 1
                           ).iloc[[idx], :].to_list()

class Sentiment_Model(nn.Module, ABC):

    def __init__(self,
                 model_name: str | None = None,
                 num_labels: int | None = None,
                 device: str = 'cpu',
                 cfg: dict | None = None) -> None:

        super().__init__()

        if isinstance(cfg, dict):
            self.__dict__.update(cfg)
        else:
            self.model_name = model_name
            self.num_labels = num_labels
            self.device = device
       
        self.mdl = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            output_hidden_states = True
        )
       
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    #TODO: add desc for torch.Tensor
    def forward(self,
                text_inputs: torch.Tensor) -> torch.Tensor:
        
        self.tokenized_inputs = self.tokenizer(
                text_inputs, 
                return_tensors="pt", 
                truncation=True, 
                padding=True
            ).to(self.device)

        outs = self.mdl(**self.tokenized_inputs)
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

                self.weights_shape = qml.QAOAEmbedding(n_layers=self.n_layers,
                                                       n_wires=len(self.wires)
                                                       )
 

    def _init_circuit(self,
                     weights: Annotated[torch.Tensor, ('n_layers', 'custom_val')] | str | None = None
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
                
                self.embed_size = 1 << len(bin(self.embed_size).split('b')[-1])

                assert bin(self.embed_size).split('b')[-1].count('1') == 1, "self.embed_size is not a power of 2"

                #here weights are for compatibility, they don't serve any meaningful purpose
                @cond_decorator(self.out, qml.qnode(self.dev, interface='torch'))
                def circuit(features: Sequence,
                            weights: None = None,
                            pad_val: complex = self.pad_val,
                            **kwargs
                            ) -> torch.Tensor:

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
                                        Annotated[
                                            Iterable[
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
                        'wires' : self.encoder.wires
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

class PatchTST_Quantum_Sentiment(nn.Module):

    def __init__(self,
                 time_series_model: nn.Module,
                 time_series_config: dict,
                 sentiment_model: nn.Module,
                 sentiment_config: dict,
                 sentiment_embed_dim: int,
                 quantum_model: nn.Module,
                 quantum_model_config: dict,
                 quantum_dim: int,
                 quantum_stride: int,
                 quantum_depth: int, #depth of the circuit in operations
                 dim_post_quantum: int,
                 max_quantum_register_size: int = 5 #max register size in qubits
                 ) -> None:

        super().__init__()

        #Time series model initialization
        self.time_series_model = time_series_model(time_series_config) #init model with config
        self.time_series_proj_dim = time_series_config.d_model #define the projection dim ,TODO: change to generalize the integration point
        

        #Sentiment model initialization
        self.sentiment_proj = nn.Linear(sentiment_embed_dim, self.time_series_proj_dim) #projection, TODO: change to generalize the integration point
        self.sentiment_model = sentiment_model(cfg=sentiment_config) #init the model with config
        
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
            self.num_registers= 0
            for _ in range((1 << self.n_qubits) - 1, quantum_dim, quantum_stride):
                num_registers += 1
        else:
            self.num_registers = 1

        #self.quantum_model.mapping holds a mapping from the names of the layers to their pennylane classes
        num_params_quantum_layer = self.quantum_model.mapping[self.quantum_model.layer_type].\
                                                                        shape(n_wires=self.n_qubits,
                                                                              n_layers=1)
        #post-quantum algorithm projection
        self.post_q_proj = nn.Linear(1 << self.n_qubits,
                                     self.time_series_proj_dim
                                     )

        self.out_proj = nn.Linear(self.time_series_proj_dim, time_series_config.prediction_length)

    def forward(self,
                time_series_inputs: Annotated[torch.Tensor, ('batch_size', 'n_time_steps', 'time_series_dim')],
                news_inputs: Annotated[torch.Tensor, ('batch_size', 'm_news_articles', 'text_len')]
                ) -> torch.Tensor:
        
        batch_size = time_series_inputs.shape[0]

        news_embeds = self.sentiment_model.forward(news_inputs).hidden_states[-1] #(batch, n_samples, text_length, embed_size)
        news_embeds_agg = news_embeds.mean(dim=1) #(batch, embed_size)
        print(news_embeds_agg.shape)
        news_embeds_proj = self.sentiment_proj(news_embeds_agg) #(batch, d_model)
        print(f'News_embeds: {news_embeds_proj.shape}')

        ts_out = self.time_series_model(time_series_inputs).hidden_states[-1] #(batch, n_patches, embed_size, d_model)
        ts_out_agg = ts_out.mean(dim=(1,2))
        print(f'Time-series embeds: {ts_out_agg.shape}')

        combined = torch.cat([ts_out_agg, news_embeds_proj], dim=1) #(batch, 2*d_model)

        if combined.shape[1] < self.quantum_dim:
            pad = torch.zeros(batch_size,
                              self.quantum_dim - combined.shape[1],
                              device=combined.device
                              )

            combined = torch.cat([combined, pad], dim=1)
        
        q_outs = [] #quantum_registers' outputs
        for idr, register in enumerate(range(0, self.num_registers, self.quantum_stride)):
            to_encode = combined[:, idr: idr+(1 << self.max_quantum_register_size)]
            to_encode = F.normalize(to_encode, p=2, dim=1)*2*np.pi #normalize to [0, 2pi] to encode
            q_outs.append(self.quantum_model.forward(to_encode))

        q_outs = torch.cat(q_outs, dim=1).float()
        print(f'q_outs: {q_outs.shape}') #(batch, )
        print(q_outs.dtype)

        post_q_outs = self.post_q_proj(q_outs)
        print(f'post_q_outs: {post_q_outs.shape}')
        out = self.out_proj(post_q_outs)
        print(f'out: {out.shape}')
        return out


if __name__ == '__main__':
    # ----------  Model parameters ----------
    batch_size = 4
    seq_len = 10               # time series sequence length
    num_features = 5
    prediction_length = 2
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


    #Sentiment model shenanigans
    sentiment_embed_dim = 768 
    sentiment_config = {
            'model_name' : "blanchefort/rubert-base-cased-sentiment".strip(),
            'num_labels' : 3,
            'device' : 'cpu'
            }


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
                            'out' : False
                            }
        }

    

    # ---------- Instantiate the combined model ----------
    combined_model = PatchTST_Quantum_Sentiment(
        time_series_model=PatchTSTForPrediction,
        time_series_config=time_series_config,
        sentiment_model=Sentiment_Model,
        sentiment_config=sentiment_config,
        sentiment_embed_dim=sentiment_embed_dim,
        quantum_model=Quantum_Kernel,
        quantum_model_config=quantum_model_config,
        quantum_dim=quantum_dim,
        quantum_stride=quantum_stride,
        quantum_depth=quantum_depth,
        dim_post_quantum=dim_post_quantum,
        max_quantum_register_size=max_quantum_register_size
    )

    print("Combined model instantiated successfully.")

    # ---------- Prepare data ----------

    path_news_data = os.path.join('..', 'data', 'news_data')
    news_data = News_Dataset(path_data)


    path_time_series_data = os.path.join('..', 'data', 'stock_data')
    time_series_inputs = torch.randn(batch_size, seq_len, num_features)

    # News input – dummy token IDs (batch, num_articles, seq_len_text)
    news_inputs = ['this is good', 'this is bad', 'nice', 'really bad']

    # ---------- Forward pass ----------
    output = combined_model(time_series_inputs, news_inputs)
    print(f"Forward output shape: {output.shape}")   # (batch, prediction_length)

    # ---------- Simple backward pass ----------
    target = torch.randn(batch_size, prediction_length)
    loss = nn.MSELoss()(output, target)
    loss.backward()
    print("Backward pass completed. Gradients exist:", any(
        p.grad is not None for p in combined_model.parameters()
    ))
