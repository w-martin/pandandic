from pathlib import Path

import pandas as pd
from pandantic import BaseFrame, Column


class FooFrame(BaseFrame):
    foo = Column(type=str)
    bar = Column(type=int)


data = FooFrame().from_csv(Path(__file__).parent.joinpath("foo.csv").as_posix())
pd.testing.assert_frame_equal(
    pd.DataFrame(
        columns=["foo", "bar"],
        data=[
            ["a", 1],
            ["b", 2],
            ["c", 3]
        ]
    )
)
