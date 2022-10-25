import datetime
from pathlib import Path

from pandandic import BaseFrame, Column, ColumnSet, ColumnGroup, DefinedLater


class ExpertFrame(BaseFrame):
    """
    Aliasing can be used to dynamically set columns or column set members at runtime.
    """
    date = Column(type=datetime.date, alias=DefinedLater)
    metadata = ColumnSet(members=DefinedLater)

    temperature = ColumnSet(type=float, members=["temperature-\d+"], regex=True)
    door_open = ColumnSet(type=bool, members=["door-open-0", "door-open-1", "door-open-2"], regex=False)

    time_series = ColumnGroup(members=[temperature, door_open])


# anything DefinedLater MUST be set before ExpertFrame reads or accesses a Column or ColumnSet via attribute
ExpertFrame.date.alias = "date"
ExpertFrame.metadata.members = ["comment", "ref"]

df = ExpertFrame().read_csv(Path(__file__).parent.joinpath("advanced.csv").as_posix())
df.set_index(ExpertFrame.date.column_name, inplace=True)  # now sets index with the defined alias
print(df.metadata)
