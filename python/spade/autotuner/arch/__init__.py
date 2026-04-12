from .arch_base import Arch
from .A100 import *
from .RTX4090 import *
from .H100 import *

AttnDevice = {
    (8,0): A100,
    (8,9): RTX4090,
    (9,0): H100,
}
