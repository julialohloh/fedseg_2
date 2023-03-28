
####################
# REQURIED MODULES #
####################

# Libs
import json

# Custom
from parsers import ModelParser
from models.relaynet.relay_net import ReLayNet
from models.hrnet.model4c import HRNetRefine
from models.hrnet.channelconfig import get_cfg_defaults
from .conftest import PackPaddedLSTM


def init_modelparser(model):
    parser = ModelParser(model)
    architecture = parser.parse(verbose=False)
    pytest_list = []
    for i in range(len(architecture)):
        pytest_list.append(architecture[i]['l_type'])
    return architecture, pytest_list

model_params = {
    'num_channels': 1,
    'num_filters': 64,
    'kernel_h': 7,
    'kernel_w': 3,
    'kernel_c': 1,
    'stride_conv': 1,
    'pool': 2,
    'stride_pool': 2,
    'num_class': 3,
    'epochs': 6
    }
def test_relaynet(relaynet_model):
    """
    C1: Check that number of layers obtained from model is consistent with layer output obtained by modelparser function
    C2: Check sequence of layers from model consistent with layer output obtained by modelparser function
    """
    model = ReLayNet(model_params)
    architecture, pytest_list = init_modelparser(model)
    assert len(architecture) == len(relaynet_model[0])
    assert pytest_list == relaynet_model[1]

def test_hrnet(hrnet_model):
    """
    C1: Check that number of layers obtained from model is consistent with layer output obtained by modelparser function
    C2: Check sequence of layers from model consistent with layer output obtained by modelparser function
    """
    configs = get_cfg_defaults()
    model = HRNetRefine(configs)
    architecture, pytest_list = init_modelparser(model)
    assert len(architecture) == len(hrnet_model[0])
    assert pytest_list == hrnet_model[1]

def test_lstmnet(LSTM_model):
    """
    C1: Check that number of layers obtained from model is consistent with layer output obtained by modelparser function
    C2: Check sequence of layers from model consistent with layer output obtained by modelparser function

    """
    model = PackPaddedLSTM()
    architecture, pytest_list = init_modelparser(model)
    assert len(architecture) == len(LSTM_model[0])
    assert pytest_list == LSTM_model[1]

def test_lstm_json(LSTM_json):
    model = PackPaddedLSTM()
    parser = ModelParser(model)
    architecture = parser.parse(verbose=False)
    out_path = "test_lstm"
    parser.export(out_path)
    with open("test_lstm/architecture.json", 'r') as f:
        LSTM_result = json.load(f)
    assert LSTM_result == LSTM_json

    