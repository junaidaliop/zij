"""First-order optimizers."""

from .a2grad import A2GradExp, A2GradInc, A2GradUni
from .accsgd import AccSGD
from .adabelief import AdaBelief
from .adabound import AdaBound, AdaBoundW
from .adagc import AdaGC
from .adai import Adai
from .adam_atan2 import AdamAtan2
from .adamod import AdaMod
from .adamp import SGDP, AdamP
from .adan import Adan
from .adanorm import AdaNorm
from .adapnm import AdaPNM
from .adashift import AdaShift
from .adasmooth import AdaSmooth
from .ademamix import AdEMAMix
from .adopt import ADOPT
from .aggmo import AggMo
from .amos import Amos
from .avagrad import AvaGrad
from .diffgrad import DiffGrad
from .exadam import EXAdam
from .fadam import FAdam
from .focus import FOCUS
from .grams import Grams
from .gravity import Gravity
from .kourkoutas_beta import KourkoutasSoftmaxFlex
from .lamb import Lamb
from .laprop import LaProp
from .lars import LARS
from .lion import Lion
from .lookahead import Lookahead
from .madgrad import MADGRAD, MirrorMADGRAD
from .mars import MARS
from .novograd import NovoGrad
from .padam import PAdam
from .pid import PID
from .qhoptim import QHM, QHAdam
from .ranger import Ranger
from .ranger21 import Ranger21
from .sgd_sai import SGDSaI
from .signsgd import SignSGD
from .stable_adamw import StableAdamW
from .swats import SWATS
from .tiger import Tiger
from .yogi import Yogi

__all__ = [
    "A2GradExp",
    "A2GradInc",
    "A2GradUni",
    "ADOPT",
    "AccSGD",
    "AdaBelief",
    "AdaBound",
    "AdaBoundW",
    "AdaGC",
    "Adai",
    "AdaMod",
    "AdamAtan2",
    "AdaNorm",
    "AdEMAMix",
    "AdamP",
    "Adan",
    "AdaPNM",
    "AdaShift",
    "AdaSmooth",
    "AggMo",
    "Amos",
    "AvaGrad",
    "DiffGrad",
    "EXAdam",
    "FAdam",
    "FOCUS",
    "Grams",
    "Gravity",
    "KourkoutasSoftmaxFlex",
    "Lamb",
    "LaProp",
    "LARS",
    "Lion",
    "Lookahead",
    "MADGRAD",
    "MARS",
    "MirrorMADGRAD",
    "NovoGrad",
    "PAdam",
    "PID",
    "SGDSaI",
    "SignSGD",
    "QHAdam",
    "QHM",
    "Ranger",
    "Ranger21",
    "SGDP",
    "StableAdamW",
    "SWATS",
    "Tiger",
    "Yogi",
]
