import datetime
from pathlib import Path

from pandandic import BaseFrame, Column, ColumnGroup


class TemperatureFrame(BaseFrame):
    """
    A ColumnGroup can use a list of column names or a regex to specify multiple columns at once.

    An exception is raised if members overlap, unless greedy_column_groups is set to True.
    In that case, the first member to match is assigned that group.

    A column group can be accessed like an attribute to provide a DataFrame view.
    """
    date = Column(type=datetime.date)
    temperature = ColumnGroup(type=float, members=["temperature-\d+"], regex=True)


df = TemperatureFrame().read_csv(Path(__file__).parent.joinpath("intermediate.csv").as_posix())
df.set_index(TemperatureFrame.date.name, inplace=True)
print(df.temperature)
