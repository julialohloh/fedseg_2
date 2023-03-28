####################
# REQURIED MODULES #
####################

# Libs
import json
import pytest
import re
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

# Custom
from models.relaynet.relay_net import ReLayNet
from models.hrnet.model4c import HRNetRefine
from models.hrnet.channelconfig import get_cfg_defaults

from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as F

################
# Model_params #
################

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

##################
# PackPaddedLSTM #
##################
class PackPaddedLSTM(nn.Module):
    """LSTM model with pack_padded layers."""

    def __init__(
        self,
        vocab_size: int = 60,
        embedding_size: int = 128,
        output_size: int = 18,
        hidden_size: int = 32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, self.hidden_size, num_layers=1)  # type: ignore[no-untyped-call] # noqa: E501
        self.hidden2out = nn.Linear(self.hidden_size, output_size)
        self.dropout_layer = nn.Dropout(p=0.2)

    def forward(self, batch: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        hidden1 = torch.ones(1, batch.size(-1), self.hidden_size, device=batch.device)
        hidden2 = torch.ones(1, batch.size(-1), self.hidden_size, device=batch.device)
        embeds = self.embedding(batch)
        packed_input = pack_padded_sequence(embeds, lengths)
        _, (ht, _) = self.lstm(packed_input, (hidden1, hidden2))
        output = self.dropout_layer(ht[-1])
        output = self.hidden2out(output)
        output = F.log_softmax(output, dim=1)
        return cast(torch.Tensor, output)


def model_output(model):
    m_list = []
    for _, m in model.named_modules():
        if len(list(m.named_modules()))==1:
            m_list.append(m)
    struc_list = []
    for i in m_list:
        module_name = re.match("^\w+",str(i)).group()
        struc_list.append(module_name)
    return m_list, struc_list


###################
# Pytest fixtures #
###################

@pytest.fixture
def relaynet_model():
    model = ReLayNet(model_params)
    m_list, struc_list = model_output(model)
    return m_list, struc_list

@pytest.fixture
def hrnet_model():
    configs = get_cfg_defaults()
    model = HRNetRefine(configs)
    m_list, struc_list = model_output(model)
    return m_list, struc_list

@pytest.fixture
def LSTM_model():
    model = PackPaddedLSTM()
    m_list, struc_list = model_output(model)
    return m_list, struc_list

@pytest.fixture
def LSTM_json():
    with open('test_LSTM.json', 'r') as f:
        data = json.load(f)
    return data
