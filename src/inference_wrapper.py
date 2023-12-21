from datetime import datetime, timedelta
from os import listdir, makedirs
from os.path import exists, getsize, join
from typing import Any, Dict, Iterable, Tuple

import numpy as np
from pandas import DataFrame, concat
from tqdm import tqdm

from inferencer import Inferencer


class InferenceWrapper:
    def __init__(self, config: Dict[str, Dict[str, Any]]) -> None:
        self.cfg = config

    def run(self) -> None:
        # init
        start = datetime.now()
        self.df_coords = DataFrame(data=None, columns=Inferencer.df_coords_columns)
        # collect image ids
        img_ids: Dict[str, Iterable[str]] = {}
        for folder in sorted(listdir(self.cfg['paths']['data'])):
            img_ids[folder] = np.unique(np.asarray(
                [str(x).split('.')[0] for x in sorted(listdir(join(self.cfg['paths']['data'], folder)))]
            ))
        # loop over tasks
        for folder, task_img_ids in tqdm(img_ids.items()):
            # init
            inferencer = Inferencer(self.cfg)
            # load data
            task_path = join(self.cfg['paths']['data'], folder)
            for img_id in task_img_ids:
                path_to_img = join(task_path, f'{img_id}.jpg')
                path_to_json = join(task_path, f'{img_id}.json')
                if exists(path_to_img) and exists(path_to_json):
                    if not getsize(path_to_json) == 0:
                        inferencer.add_data_item(path_to_img, path_to_json)
            # predict
            inferencer.predict()
            self.df_coords = concat((self.df_coords, inferencer.get_coords()), axis=0, ignore_index=True)
        # calculate metrics
        self.acc, self.avg_dist = self._calculate_metrics(self.df_coords)
        self.wall = datetime.now() - start

    def save_coords(self) -> None:
        makedirs(self.cfg['paths']['results'], exist_ok=True)
        path_to_df = join(self.cfg['paths']['results'], 'coords.csv')
        self.df_coords.to_csv(path_to_df, header=True)

    def save_metrics(self) -> Iterable[str]:
        "Returns the saved lines as a list of strings"
        metrics_str = [
            f'Accuracy:\t{self.acc}\n',
            f'Avg. distance:\t{self.avg_dist}\n',
            f'Wall time:\t{str(self.wall)[:-7]}\n',
        ]
        makedirs(self.cfg['paths']['results'], exist_ok=True)
        path_to_metrics = join(self.cfg['paths']['results'], 'metrics.txt')
        with open(path_to_metrics, 'w') as f:
            f.writelines(metrics_str)
        return metrics_str

    @property
    def coords(self) -> DataFrame:
        return self.df_coords

    @property
    def accuracy(self) -> float:
        return self.acc

    @property
    def avg_distance(self) -> float:
        return self.avg_dist

    @property
    def wall_time(self) -> timedelta:
        return self.wall

    def _calculate_metrics(self, df: DataFrame) -> Tuple[float, float]:
        """
        Return:
            acc, avg_dist
        """
        df.loc[:, 'dist'] = np.hypot(df['true_x'] - df['pred_x'], df['true_y'] - df['pred_y'])
        df.loc[:, 'dist_bool'] = df['dist'] < float(self.cfg['metrics']['accuracy_threshold'])
        acc = df['dist_bool'].sum() / len(df)
        avg_dist = df['dist'].mean()
        return acc, avg_dist
