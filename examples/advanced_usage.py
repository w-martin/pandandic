import datetime
from pathlib import Path

from pandandic import BaseFrame, Column, ColumnGroup
from pandandic.group import Group


class AdvancedFrame(BaseFrame):
    """
    A Group can be used to group together multiple column groups and columns.
    It can be accessed like an attribute to provide a dataframe view.
    """
    date = Column(type=datetime.date)
    temperature = ColumnGroup(type=float, members=["temperature-\d+"], regex=True)
    door_open = ColumnGroup(type=bool, members=["door-open-0", "door-open-1", "door-open-2"], regex=False)
    ref = Column(type=int)
    comment = Column(type=str)

    numerical = Group(members=[temperature, ref])
    time_series = Group(members=[temperature, door_open])


df = AdvancedFrame().read_csv(Path(__file__).parent.joinpath("advanced.csv").as_posix())
df.set_index(AdvancedFrame.date.name, inplace=True)
print(df.time_series)
