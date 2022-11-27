import sys

import pandas as pd

from .base_schema import BaseSchema

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


class BaseFrame(pd.DataFrame, BaseSchema):
    """
    Enabled schema-in-code through subclassing of pd.DataFrame.
    Wraps pandas read functions in order to apply the defined schema.
    See https://github.com/w-martin/pandandic for usage documentation or to submit bug reports.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        BaseSchema.__init__(self)

    def __getattribute__(self, item):
        if item.startswith("_"):
            return super().__getattribute__(item)

        item_is_a_column = any(
            map(lambda m: item in m, filter(None, (self._column_map, self._column_set_map, self._column_group_map))))
        if item_is_a_column:
            columns = BaseSchema.get_columns(self, item)
            if 1 == len(columns):
                columns = columns[0]
            return self[columns]
        return super().__getattribute__(item)

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self)

    def from_csv(self, *args, **kwargs) -> Self:
        df = BaseSchema.read_csv(self, *args, **kwargs)
        pd.DataFrame.__init__(self, df)
        return self

    def from_excel(self, *args, **kwargs) -> Self:
        df = BaseFrame.read_excel(self, *args, **kwargs)
        super().__init__(df)
        return self

    def from_parquet(self, *args, **kwargs) -> Self:
        df = BaseSchema.read_parquet(self, *args, **kwargs)
        super().__init__(df)
        return self

    def from_avro(self, *args, **kwargs) -> Self:
        df = BaseSchema.read_avro(self, *args, **kwargs)
        super().__init__(df)
        return self

    def from_df(self, df) -> Self:
        df = BaseSchema.apply(self, df)
        super().__init__(df)
        return self
