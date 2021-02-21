import os
import urllib.request

_data_name = 'data.csv'

_data_root = 'data'

_data_path = 'https://raw.githubusercontent.com/tensorflow/tfx/master/tfx/examples/chicago_taxi_pipeline/data/simple/data.csv'
if __name__ == '__main__':
    urllib.request.urlretrieve(
        _data_path,
        os.path.join(_data_root, _data_name),
    )
