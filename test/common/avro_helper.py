import pandas as pd


def to_avro(df, filename, *args, **kwargs):
    if "index" in kwargs:
        del kwargs["index"]
    from pandavro import to_avro
    avro_type_dict = {
        str: 'string',
        object: 'string',
        type(None): 'string',
        int: 'int',
        float: 'double',
        pd._libs.tslibs.timestamps.Timestamp: {'type': 'long', 'logicalType': 'timestamp-micros'}
    }
    schema = {'type': 'record', 'name': 'Root',
              'fields': [{'name': k, 'type': ['null', avro_type_dict[type(v)]]} for k, v in
                         df.iloc[0].to_dict().items()]}
    for column, dtype in df.dtypes.items():
        if str(dtype) == "datetime64[ns]":
            df = df.copy()
            df.loc[:, column] = df.loc[:, column].dt.tz_localize("utc")
    to_avro(filename, df, *args, schema=schema, **kwargs)
