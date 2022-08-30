import numpy as np
import pandas as pd

from src.base_frame import BaseFrame
from src.column import Column

data = pd.read_csv("data.csv")


class Request(BaseModel):
    size: int
    shape: str
    name: str

a = Request(name="fred")
assert a.name == "fred"





class StarData(BaseFrame):
    star = Column(type=str)
    x = Column(type=float).with_condition(GreaterThan(5))
    y = Column(type=float)
    z = Column(type=float)
    brightness = Column(type=float)


star_data = StarData.read_csv("data.csv", allow_extra_cols=True)
first_euclidean_distance = np.sqrt((data.iloc[0].x - data.iloc[1].x))
print(first_euclidean_distance)
