# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pyscf import gto

from lapnet import base_config
from jaqmc.pp.pp_config import get_config as get_ecp_config

def get_config(input_str):
    symbol, spin = input_str.split(',')
    spin = int(spin)

    cfg = base_config.default()
    cfg['ecp'] = get_ecp_config()
    mol = gto.Mole()
    # Set up molecule
    mol.build(
        atom=f'{symbol} 0 0 0',
        basis={symbol: 'ccecpccpvdz'},
        ecp={symbol: 'ccecp'},
        spin=spin)

    cfg.system.pyscf_mol = mol
    return cfg