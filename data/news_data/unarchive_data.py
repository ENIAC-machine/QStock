from corus import load_lenta, load_lenta2, load_mokoron, load_buriy_news, load_buriy_webhose
import os
import sys
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from itertools import islice

from typing import Generator, Callable

DATA_DIR = os.path.join('.')


func_to_data = {load_lenta: ['lenta-ru-news.csv.gz'],
                load_lenta2 : ['lenta-ru-news.csv.bz2'],
                load_mokoron : ['db.sql'],
                load_buriy_news : ['news-articles-2014.tar.bz2',
                                   'news-articles-2015-part1.tar.bz2',
                                   'news-articles-2015-part2.tar.bz2'],
                load_buriy_webhose : ['webhose-2016.tar.bz2'],
                pd.read_csv : [file for file in os.listdir(DATA_DIR) if file.endswith('.csv')]
                }

def prepare_data(data_path: str | Path,
                 load_func: Callable,
                 chunk_size: int = 10_000,
                 verbose: bool = True
                 ) -> Generator[pd.DataFrame | None]:

    '''
    Function to prepare data of a single file (archive of csv) into a pd.DataFrame object

    Inputs:
        data_path: str | Path - path to the file
        load_func: Callable - function to load it with
        chunk_size: int = 10_000 - size of a chunk to process at a time
        verbose: bool = True - verbosity flag

    Outputs:
        pd.DataFrame - chunk of the file

    '''

    if data_path.endswith('.csv'):
        reader = load_func(data_path, chunksize=chunk_size)
        for df in tqdm(reader,
                             desc='Loading DataFrame chunks',
                             leave=False,
                             disable=not verbose,
                             unit=' chunks'):
            yield df


    elif data_path.split('.')[-1] in ['gz', 'bz2', 'sql']:

        #make the object an iterable for the `islice` function to work
        gen = iter(load_func(data_path))
        
        lines_cnt = 0
        with tqdm(unit=' lines', leave=False, disable=not verbose) as progress:
            while True: #because we don't know the num of elements in the generator
                data = list(islice(gen, chunk_size))
                
                if data is None or len(data) == 0:
                    break

                columns = [col for col in data[-1].__attributes__
                                    if col in ['date', 'text', 'timestamp']
                           ]

                yield pd.DataFrame(data, columns=data[-1].__attributes__)[columns]
                lines_cnt += chunk_size
                progress.update(chunk_size) 

    else:
        print(f'Unknown datatype to process: {data_path.split(".")}')
        yield None

    
def save_to_csv(data_dir:str | Path = DATA_DIR,
                slice_size: int = 10_000,
                delete_old: bool = True,
                unified_file_nm: str = 'all_data.csv',
                verbose: bool = True
                ) -> None:
    '''
    Opens archives and saves their contents in a single file

    Inputs:
        data_dir:str | Path = DATA_DIR - inputs in the format (function_to_process_file, file_names)
        slice_size: int = 10_000 - size of a chunk DataFrame to load from each archive
        delete_old: bool = True - flag to delete the old file
        unified_file_nm: str = 'all_data.csv' - name of the new unified file
        verbose: bool = True - verbosity flag

    Outputs:
        None

    '''

    path_save = os.path.join(data_dir, unified_file_nm)

    if os.path.exists(path_save) and delete_old:
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
            data_paths[idx] = os.path.join(data_dir, filenm) #integrate full path with filename
        
        data_path = data_paths[0]

        for data_path in tqdm(data_paths,
                              desc=f'Loading archive {data_path}',
                              leave=False,
                              unit=" files",
                              disable=not verbose):

            for df in prepare_data(data_path, load_func, slice_size, verbose):

                if 'date' not in df.columns:
                    df['date'] = pd.to_datetime(df['timestamp']).apply(lambda x: x.date())
                    
                df.to_csv(path_save,
                          header=header,
                          mode='a',
                          columns=['date', 'text']
                          )

    return None 

save_to_csv()
