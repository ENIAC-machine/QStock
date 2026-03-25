import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


from itertools import islice
from corus import load_lenta, load_lenta2, load_mokoron, load_buriy_news, load_buriy_webhose
from torch.utils.data import DataLoader, Dataset

from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification, AdamW,
        PatchTSTConfig, PatchTSTForPrediction
        )

from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from pathlib import Path
from typing import Callable, Generator, Any, Iterable, Annotated
from abc import abstractmethod, ABCMeta

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


class Abstract_Fin_Dataset(Dataset, ABCMeta):

    def __init__(self,
                 data_dir: str | Path = os.path.join('..', 'data'),
                 batch_size: int | None = None,
                 unified_file_nm: str = 'all_data.csv',
                 load_from_file: bool = False,
                 delete_old: bool = True,
                 ) -> None:

        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_elements = None
        self.unified_file_nm = unified_file_nm
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

    def __iter__(self) -> list[int]:
        for idx in range(0, self.num_elements, self.batch_size):
            yield list(range(idx, min(idx+self.batch_size, self.num_elements)))


class News_Dataset(Abstract_Fin_Dataset):

    def __init__(self,
                 data_dir: str | Path = os.path.join('.'),
                 slice_size: int = 10_000,
                 unified_file_nm: str = 'all_data.csv',
                 load_from_file: bool = False,
                 delete_old: bool = True,
                 **kwargs
                 ) -> None:

        super().__init__()

        self.slice_size = slice_size
        self.unified_file_nm = unified_file_nm
        self.delete_old = delete_old

        self.load()


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
            self.unified_file_nm: str = 'all_data.csv' - name of the new unified file
            verbose: bool = True - verbosity flag

        Outputs:
            None

        '''

        path_save = os.path.join(self.data_dir, self.unified_file_nm)

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

                for df in self._prepare_data(data_path, load_func, self.slice_size, verbose):

                    if 'date' not in df.columns:
                        df['date'] = pd.to_datetime(df['timestamp']).apply(lambda x: x.date())
                        
                    df.to_csv(path_save,
                              header=header,
                              mode='a',
                              columns=['date', 'text']
                              )

        return None 


    def __getitem__(self, idx: int | Iterable[int]) -> list[Any]:
        return pd.read_csv(self.unified_file_nm, skip_rows=idx[0], nrows=len(idx)).iloc[[idx], :].to_list()

class Abstract_Sentiment_Model(nn.Module, ABCMeta):

    def __init__(self,
                 model_name: str,
                 num_labels: int,
                 *args, **kwargs) -> None:

        super().__init__()

        self.model_name = model_name
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
       
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    #no shit Sherlock
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        ...

class Russian_Sentiment_Model(Abstract_Sentiment_Model):
    
    def __init__(self,
                 model_name: str = "blanchefort/rubert-base-cased-sentiment",
                 num_labels: int = 3
                 ) -> None:
        '''
        Inputs:
            model_name: str - name of the model to load
            num_labels: int - number of labels to expect from the model, default is 3,
                which corresponds to positive-neutral-negative

        '''

        super().__init__()
        
        def forward(self,
                    input_ids: Iterable[int],
                    attention_mask: torch.Tensor,
                    labels=None
                    ) -> torch.Tensor:
        
        '''
        Forward pass. If labels are provided, returns loss and logits.
        Otherwise returns only logits.
        '''

        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        if labels is not None:
            return outputs.loss, outputs.logits
        else:
            return outputs.logits

    def predict(self, texts: Iterable[str], device: str = 'cpu') -> torch.Tensor:
        '''
        Run inference on a list of texts.
        Returns predicted class indices and probabilities.
        '''

        self.eval()
        with torch.no_grad():
            encodings = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors='pt'
            ).to(device)
            logits = self.forward(encodings['input_ids'], encodings['attention_mask'])
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        return preds, probs

class Quantum_Encoder(nn.Module):

    def __init__(self,
                 embed_size: int,
                 encoder_type: str = 'Amplitude',
                 wires: Iterable | None = None,
                 device: str = 'default.qubit',
                 pad_val: int | float | complex = 0,
                 out: bool = False) -> None:

        '''
        Wrapper class for encoding via pennylane functions

        Inputs:
            features: Sequence - classical feature values to encode
            embed_size: int - size of the output embedding vector
            encoder_type: str - type of encoding scheme, currently supported are 'Amplitude' for amplitude encoding, 'Phase' for phase encoding and 'QAOA' for the encdoing strategy inspired by the QAOA
            wires: Iterable | None - wires (qubits) to encode in the quantum circuit
            pad_val: int | float | complex - value to pad the classical vector with
        Outputs:
            None

        '''

        #TODO: add assert statements

        super().__init__()

        #initialize the simulator
        self.wires = wires
        self.device = device
        self.dev = qml.device(self.device, wires=self.wires)
        
        self.encoder_type = encoder_type
        self.embed_size = embed_size
        self.pad_val = pad_val
        self.out = out
        self.circuit = None

    def _init_circuit(self,
                     weights: Sequence | None = None
                     ) -> Callable:
        ''' 
        
        Initialize the circuit for the embedding
        
        Inputs:
            None

        Outputs:
            circuit
        
        '''

        pad_features = lambda features, wires, pad_val: list(features) + [pad_val]*(len(wires) - len(features))
    

        match self.encoder_type:

            case 'Amplitude':
                
                self.embed_size = 1 << len(bin(self.embed_size).split('b')[-1])

                assert bin(self.embed_size).split('b')[-1].count('1') == 1, "self.embed_size is not a power of 2"


                #here weights are for compatibility, they don't serve any meaningful purpose
                if self.out:

                    @qml.qnode(self.dev, interface='torch')
                    def circuit(features: Sequence,
                                weights: None = None,
                                pad_val: int | float | complex = self.pad_val,
                                **kwargs
                                ) -> np.ndarray:

                        qml.AmplitudeEmbedding(features,
                                               wires=self.wires,
                                               pad_with=pad_val,
                                               **kwargs
                                               )
                        return qml.state()
                else:
                    def circuit(features: Sequence,
                                weights: None = None,
                                pad_val: int | float | complex = self.pad_val,
                                **kwargs
                                ) -> np.ndarray:

                        qml.AmplitudeEmbedding(features,
                                               wires=self.wires,
                                               pad_with=pad_val,
                                               **kwargs
                                               )
            case 'Phase':
               
                if self.out:
                    @qml.qnode(self.dev, interface='torch')
                    def circuit(features: Sequence,
                                weights: str = 'Z', #consider it a hyperparameter
                                ) -> np.ndarray:

                        features = pad_features(features, self.wires, self.pad_val)
                        qml.AngleEmbedding(features, self.wires, rotation=weights)
                        
                        return qml.state()
                
                else:
                    def circuit(features: Sequence,
                                weights: str = 'Z', #consider it a hyperparameter
                                ) -> np.ndarray:

                        features = pad_features(features, self.wires, self.pad_val)
                        qml.AngleEmbedding(features, self.wires, rotation=weights)
                     

            case 'QAOA':

                if self.out:

                    @qml.qnode(self.dev, interface='torch')
                    def circuit(features : Sequence,
                                weights: Sequence | None = weights, #here weights are actually trainable params
                                **kwargs
                                ) -> np.ndarray:

                    
                        weights = nn.Parameter(weights)
                        features = pad_features(features, self.wires, self.pad_val) 
                        qml.QAOAEmbedding(features, weights, self.wires, **kwargs)

                        return qml.state()

                else:
                    def circuit(features : Sequence,
                                weights: Sequence | None = weights, #here weights are actually trainable params
                                **kwargs
                                ) -> np.ndarray:

                    
                        weights = nn.Parameter(weights)
                        features = pad_features(features, self.wires, self.pad_val) 
                        qml.QAOAEmbedding(features, weights, self.wires, **kwargs)


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
                 layer_type : str = 'SimplifiedTwoDesign',
                 wires: Iterable = range(5),
                 layer_config : dict[str : Any] = None,
                 encoder: Quantum_Encoder = None,
                 device: str = 'default.qubit',
                 uncorr_wires: tuple | Iterable[int] = ()
                 ) -> None:

        super().__init__()
        self.layer_type = layer_type
        self.wires = wires
        self.uncorr_wires = uncorr_wires
        self.device = device

        self.encoder = encoder
        
        self.dev = qml.device(self.device, wires=self.wires) 

        if layer_config is None:
            layer_config = {'weights' : nn.Parameter(
                                            torch.Tensor(
                                                np.random.rand(
                                                    1, len(self.encoder.wires)-1, 2
                                                    )
                                                )
                                            ),
                            'initial_layer_weights' : nn.Parameter(
                                                        torch.Tensor(
                                                            np.random.rand(
                                                                len(self.encoder.wires)
                                                                )
                                                            )
                                                        ),
                            'wires' : self.encoder.wires
                            }

        self.layer_config = layer_config 

        self.mapping = {'SimplifiedTwoDesign' : qml.SimplifiedTwoDesign,
                        'StronglyEntangling' : qml.StronglyEntanglingLayers}


    def _init_circuit(self) -> Callable:

        @qml.qnode(self.dev, interface='torch')
        def circuit(features: Sequence) -> np.ndarray | torch.Tensor:
            corr_features, uncorr_features = [], []

            for idx, feature in enumerate(features):
                if idx in self.uncorr_wires:
                    uncorr_features.append(feature)
                else:
                    corr_features.append(feature)
        
            self.encoder._init_circuit()
            self.encoder.circuit(corr_features)
            self.mapping[self.layer_type](**self.layer_config)
                        
            for wire, feature in zip(self.uncorr_wires, uncorr_features):
                qml.RX(feature, wires=wire)
            

            return qml.state()


        self.circuit = circuit

        return self.circuit

    def forward(self,
                features: Sequence
                ) -> torch.Tensor:
    
        if not self.encoder.circuit:
            print('Initializing without encoder...')

        if not hasattr(self, 'circuit'):
            self._init_circuit()

        return self.circuit(features)

#TODO: add modules for time series data processing, uniting data and the main

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

        self.time_series_model = time_series_model(time_series_config)

        self.time_series_proj_dim = time_series_config.d_model
        self.sentiment_proj = nn.Linear(sentiment_embed_dim, self.time_series_proj_dim) #TODO: change to generalize the integration point

        self.sentiment_model = sentiment_model(sentiment_config)
        self.quantum_model = quantum_model(quantum_model_config)

        #compare the number of qubits needed to fully encompass the embed dim vs the max allowed register size
        self.n_qubits = min((1 << quantum_dim.bit_length()).bit_length() - 1, max_quantum_register_size)
        if self.n_qubits == max_quantum_register_size:
            num_registers = 0
            for _ in range((1 << self.n_qubits) - 1, quantum_dim, quantum_stride):
                num_registers += 1
        else:
            num_registers = 1

        num_params_quantum_layer = np.prod(quantum_model_config['layer_type'].shape(n_wires=self.n_qubits))

        #tobe refactored
        self.q_params = nn.Parameter(torch.randn(num_params_quantum_layer,
                                                 self.n_qubits,
                                                 quantum_model_config['n_layers']
                                                 )
                                     )

        #post-quantum algorithm projection
        self.post_proj = nn.Linear(num_registers * (1 << self.n_qubits), self.time_series_proj_dim)

        self.out_proj = nn.Linear(self.time_series_proj_dim, time_series_config.prediction_length)

    def forward(self,
                time_series_inputs: Annotated[torch.Tensor, 'batch_size', 'n_time_steps', 'time_series_dim']
                news_inputs: Annotated[torch.Tensor, 'batch_size', 'm_news_articles', 1]
                ) -> torch.Tensor:
        
        news_embeds_agg = self.sentiment_model(news_inputs).last_hidden_state.mean(dim=-1)
        print(news_embeds_agg.shape)

        time_series_out = self.time_series_model(time_series).last_hidden_state #(batch, n_patches, d_model)
        print(time_series_out.shape)


news = ['This is good!', 'This is bad!', 'Hello there']
time_series = []

PatchTST_Quantum_Sentiment()

