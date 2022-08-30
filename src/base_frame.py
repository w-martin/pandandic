from typing import Dict

import pandas as pd
from pandas import DataFrame

from src.column import Column


class BaseFrame(DataFrame):

    @classmethod
    def get_typed_columns(cls) -> Dict[str, Column]:
        return dict(filter(lambda kv: isinstance(kv[1], Column), cls.__dict__.items()))

    @classmethod
    def from_csv(cls, *args, **kwargs) -> DataFrame:
        kwargs["usecols"] = []
        df = pd.read_csv(*args, **kwargs)
        return df
