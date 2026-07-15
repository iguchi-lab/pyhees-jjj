import pytest
from jjjexperiment.result import *
import collections.abc

from os import path

# NOTE: それぞれ用意する必要がある理由
# 各方式ごとにjsonの内容が包含でなく排他的であり 'TYPE'の書替のみでは不可能であるため
INPUT_SAMPLE_TYPE1_PATH = path.join(path.dirname(__file__), 'input_sample_type1.json')
INPUT_SAMPLE_TYPE2_PATH = path.join(path.dirname(__file__), 'input_sample_type2.json')
INPUT_SAMPLE_TYPE3_PATH = path.join(path.dirname(__file__), 'input_sample_type3.json')
INPUT_SAMPLE_TYPE4_PATH = path.join(path.dirname(__file__), 'input_sample_type4.json')

"""テストインプットデータの差し替え設定エリア"""

# V_supply_d_t_i の設計風量によるキャップロジック(案1,案2) のテストに有用なデータ
# 委員の方が検証のために使用されていたデータをいただいたものです

# INPUT_SAMPLE_TYPE1_PATH = path.join(path.dirname(__file__), 'input_sample_vsup_cap_logics.json')

@pytest.fixture
def expected_inputs():
    """ テストコードは下記の入力を想定したものになっています """
    inputs = SutValues(
        q_rtd_C = 5600,
        q_rtd_H = 6685.3,
        q_max_C = 5944.619999999999,
        q_max_H = 10047.047813999998,
        e_rtd_C = 2.8512,
        e_rtd_H = 3.855424,
    )
    return inputs

@pytest.fixture
def expected_result_type1():
    """ 上記の入力内容で期待される結果 """
    return ResultSummary(E_C=17982.693457290967, E_H=36558.649546681496)

@pytest.fixture
def expected_result_type2():
    """ 上記の入力内容で期待される結果 """
    return ResultSummary(E_C=16631.770301700133, E_H=40692.8527046848)

def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            r = deep_update(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d

def load_input_yaml(path: str) -> dict:
    import yaml
    with open(path, 'r', encoding='utf-8') as f:
        input_yaml = yaml.safe_load(f)
    return input_yaml
