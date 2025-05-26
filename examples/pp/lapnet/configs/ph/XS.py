
# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pyscf import gto

import numpy as np

from lapnet import base_config
from jaqmc.pp.ph.data import PH_config
from jaqmc.pp.pp_config import get_config as get_ecp_config

@PH_config
def get_config(input_str):
    symbol, dist, unit, spin,Xup,Xdn,Yup,Ydn = input_str.split(',')

    # Get default options.
    cfg = base_config.default()
    cfg['ecp'] = get_ecp_config()

    mol = gto.Mole()

    # Set up molecule
    mol.build(
        atom=f'{symbol} 0 0 0; S 0 0 {dist}',
        basis={symbol: 'ccecpccpvdz', 'S': 'ccecpccpvdz'},
        ecp={symbol: 'ccecp', 'S': 'ccecp'},
        spin=int(spin), unit=unit)

    cfg.system.pyscf_mol = mol
    cfg.system.atom_spin_configs = [(int(Xup), int(Xdn)), (int(Yup), int(Ydn))]
    cfg.ecp.ph_elements = (symbol, 'S')
    return cfg