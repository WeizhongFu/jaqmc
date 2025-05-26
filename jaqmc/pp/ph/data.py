# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pathlib

# value: spin
PH_TM_ELEMENTS = {'Cr': 6,
                  'Mn': 5,
                  'Fe': 4,
                  'Co': 3,
                  'Ni': 2,
                  'Cu': 1,
                  'Zn': 0}
# value: (charge, use_hf)
TM_ELEMENTS_INFO = {'Cr': (14.0, False),
                    'Mn': (15.0, True),
                    'Fe': (16.0, False),
                    'Co': (17.0, False),
                    'Ni': (18.0, True),
                    'Cu': (19.0, True),
                    'Zn': (20.0, False)}

# value: spin
PH_MG_ELEMENTS = {'S': 2}
# value: (charge, use_hf)
MG_ELEMENTS_INFO = {'S': (6.0, False)}

PH_ELEMENTS = {**PH_TM_ELEMENTS, **PH_MG_ELEMENTS}

def PH_config(get_config,
              get_ecp_cfg_ref_from_cfg=lambda x: x.ecp,
              get_pyscf_mol_from_cfg=lambda x: x.system.pyscf_mol):
    def wrapper(*args):
        cfg = get_config(*args)
        pyscf_mol = get_pyscf_mol_from_cfg(cfg)
        ecp_cfg_ref = get_ecp_cfg_ref_from_cfg(cfg)
        ecp_cfg_ref.ph_info = gen_ph_info(pyscf_mol._atom,
                                          ph_elements=ecp_cfg_ref.ph_elements)
        return cfg
    return wrapper

def parse_TM_xml(xml_file):
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_file)
    root = tree.getroot()

    def get_data_arr(index):
        data = [float(y) for x in root[2][index][0][1].text.split('\n') for y in x.strip().split(' ') if y != '']
        return np.array(data)
    s_arr = get_data_arr(0)
    p_arr = get_data_arr(1)
    d_arr = get_data_arr(2)

    local_nl = d_arr
    v0_nl = s_arr - local_nl
    v1_nl = p_arr - local_nl

    # This relation should hold: 2 * v0_nl == 3 * v1_nl
    return local_nl + v0_nl, - v0_nl / 6

def gen_ph_data_TM(element):
    charge, use_hf = TM_ELEMENTS_INFO[element]
    filename = f'{element}.{"hf" if use_hf else "cc"}.xml'
    xml_file = pathlib.Path(__file__).parent.resolve() / 'raw_data' / 'TM' / filename
    loc_data, l2_data = parse_TM_xml(str(xml_file))

    # The data in XML is in form:
    #
    # l2_data  = r * v_{L^2}
    # loc_data + charge = r * \tilde{v}_loc
    return loc_data + charge, l2_data

def gen_ph_data_MG(element):
    charge, use_hf = MG_ELEMENTS_INFO[element]
    filename = f'{element}.{"hf" if use_hf else "cc"}.xml'
    xml_file = pathlib.Path(__file__).parent.resolve() / 'raw_data' / 'MG' / filename
    loc_data, l2_data = parse_TM_xml(str(xml_file))

    # The data in XML is in form:
    #
    # l2_data  = r * v_{L^2}
    # loc_data + charge = r * \tilde{v}_loc
    return loc_data + charge, l2_data

def gen_ph_data(element):
    if element in PH_TM_ELEMENTS:
        return gen_ph_data_TM(element)
    elif element in PH_MG_ELEMENTS:
        return gen_ph_data_MG(element)
    else:
        raise NotImplementedError(f'The PH for {element} is not supported yet')

def gen_ph_info(atoms, ph_elements=None):
    def should_consider_ph(element):
        # Try all available PH elements
        if ph_elements is None:
            return element in PH_ELEMENTS

        return element in ph_elements

    ph_atom_pos = []
    ph_data = dict()
    for symbol, pos in atoms:
        if should_consider_ph(symbol):
            ph_atom_pos.append((symbol, pos))
            if symbol not in ph_data:
                ph_data[symbol] = gen_ph_data(symbol)
    return (ph_atom_pos, ph_data)

def gen_ph_data_Co():
    '''
    This function generate Pseudo-Hamiltonian corresponding to
    https://pubs.acs.org/doi/10.1021/acs.jctc.1c00992
    , which only support element Co.

    This PH is deprecated due to the presence of
    https://pubs.aip.org/aip/jcp/article/159/16/164114/2918607/Locality-error-free-effective-core-potentials-for
    , which support all the first row transition metals, including Co.
    '''
    def parse_xml(xml_file):
        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_file)
        root = tree.getroot()
        loc_data = [float(y) for x in root[3][0][0][1].text.split('\n') for y in x.strip().split(' ') if y != '']
        L2_data = [float(y) for x in root[2][0][1].text.split('\n') for y in x.strip().split(' ') if y != '']
        return np.array(loc_data), np.array(L2_data)

    xml_file = pathlib.Path(__file__).parent.resolve() / 'raw_data' / 'TM' / 'Co.pure.xml'
    loc_data, l2_data = parse_xml(str(xml_file))

    # The data in XML is in form:
    #
    # l2_data  = r * v_{L^2}
    # loc_data + 17 = r * \tilde{v}_loc
    #
    # Here `l2_data` and `loc_data` are the data in "L2 section" and s-channel
    # in "semilocal section" in the XML file respectively.
    return loc_data + 17.0, l2_data
