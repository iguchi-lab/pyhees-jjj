import pytest
import json
import copy

from jjjexperiment.main import calc
from jjjexperiment.logger import LimitedLoggerAdapter as _logger
from jjjexperiment.options import *

from test_utils.utils import *

from jjjexperiment.di_container import *
from injector import Injector

# NOTE: DIコンテナライブラリ Injector 導入の目的:
# 現時点では、出力用データフレームの操作のみ、行っているが、
# 将来的には、建研さまロジックのカスタマイズによる
# 引数・返り値のいくつもの追加を解消するため

class Test_DIコンテナ:

    @pytest.mark.skip(reason="本ロジックとは無関係のため")
    def test_Injector(self):
        """ DIコンテナの挙動テスト """
        di = Injector(JJJExperimentModule())
        df_holder = di.get(UfVarsDataFrame)
        df_holder.update_df({'x':[1,2,3], 'y':[2,3,4]})
        df_holder.export_to_csv("di_test.csv")
