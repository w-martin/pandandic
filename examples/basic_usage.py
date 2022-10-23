from pathlib import Path

from pandandic import BaseFrame, Column


class FooFrame(BaseFrame):
    """
    Each column set below will be read with the given type. Columns can be accessed like attributes to return Series
    slices in the usual way.
    """
    foo = Column(type=str)
    bar = Column(type=int)


path = Path(__file__).parent.joinpath("basic.csv").as_posix()
data = FooFrame().read_csv(path)
print(data.foo)
print(data.bar)
