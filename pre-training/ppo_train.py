import numpy as np
import torch
import argparse
from pprint import pformat
from torch.distributions import Categorical
import torch.nn.functional as F
from policynet import *

switcher = {'gcn':PolicyNet,'mlp':PolicyNet2}
print(switcher)