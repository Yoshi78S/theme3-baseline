from model.bsarec import BSARecModel
from model.caser import CaserModel
from model.gru4rec import GRU4RecModel
from model.sasrec import SASRecModel
from model.bert4rec import BERT4RecModel
from model.fmlprec import FMLPRecModel
from model.duorec import DuoRecModel
from model.fearec import FEARecModel
from model.lrurec import LRURecModel
from model.mamba4rec import Mamba4RecModel
from model.icsrec import ICSRecModel
from model.iclrec import ICLRecModel
from model.sigma import SIGMAModel

MODEL_DICT = {
    "bsarec": BSARecModel,
    "caser": CaserModel,
    "gru4rec": GRU4RecModel,
    "sasrec": SASRecModel,
    "bert4rec": BERT4RecModel,
    "fmlprec": FMLPRecModel,
    "duorec": DuoRecModel,
    "fearec": FEARecModel,
    "lrurec": LRURecModel,
    "mamba4rec": Mamba4RecModel,
    "icsrec": ICSRecModel,
    "iclrec": ICLRecModel,
    "sigma": SIGMAModel,
}
