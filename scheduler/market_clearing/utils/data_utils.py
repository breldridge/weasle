# load JSON data to GDX

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os, sys, json, csv
import glob
from os.path import join
import copy
import gams
import pandas as pd
import numpy as np
import logging
import datetime
from market_clearing.utils import format_library as fl
import matplotlib.pyplot as plt

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
grandparent_directory = os.path.abspath(os.path.join(parent_directory, os.pardir))
if parent_directory not in sys.path:
    sys.path.append(parent_directory)
if grandparent_directory not in sys.path:
    sys.path.append(grandparent_directory)

from market_clearing.offer_data import dummy_data
from .data_types import gams_symbols, storage_attributes, parameter_domains

class GamsTupleEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, tuple):
            return {'__tuple__': True, 'items': [str(item) for item in obj]}
        return json.JSONEncoder.default(self, obj)

def decode_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()

def encode_tuple(obj):
    key_list = []
    val_list = []
    for rec in obj:
        key_list.append(tuple(rec.keys))
        val_list.append(rec.value)
    return {'__tuple__': True, 'keys': key_list, 'values': val_list}

def decode_tuple(encoded_tuple):
    if '__tuple__' in encoded_tuple and encoded_tuple['__tuple__']:
        keys = encoded_tuple['keys']
        values = encoded_tuple['values']
        obj = []
        for i in range(len(keys)):
            obj.append({'keys': keys[i], 'value': values[i]})
        return obj
    else:
        raise ValueError("Invalid input. The encoded tuple must have '__tuple__' set to True.")

def fill_if_blank(this_dict, key, blank):
    if key not in this_dict.keys():
        this_dict[key] = blank
    return this_dict

def update_nested_list(this_dict, klist, vlist):
    ''' Updates a list in a nested dictionary, creating keys along the way if needed.
    '''
    nkeys = len(klist)
    if nkeys > 3:
        raise ValueError(f"No rule make for {nkeys} keys. Max enabled is 3 keys.")
    try:
        if nkeys == 3:
            this_dict[klist[0]][klist[1]][klist[2]] += vlist
        elif nkeys == 2:
            this_dict[klist[0]][klist[1]] += vlist
        elif nkeys == 1:
            this_dict[klist[0]] += vlist
    except KeyError:
        blank = {}
        for i in range(len(klist)):
            if i+1 == nkeys:
                blank = []
            if i == 0:
                this_dict = fill_if_blank(this_dict, klist[i], blank)
            elif i == 1:
                i1_dict = copy.copy(this_dict[klist[0]])
                i1_dict = fill_if_blank(i1_dict, klist[i], blank)
                this_dict[klist[0]] = i1_dict
            elif i == 2:
                i2_dict = copy.copy(this_dict[klist[0]][klist[1]])
                i2_dict = fill_if_blank(i2_dict, klist[i], blank)
                this_dict[klist[0]][klist[1]] = i2_dict
        this_dict = update_nested_list(this_dict, klist, vlist)
    return this_dict

def uid_to_time(uid, to_datetime=False, return_mkt_spec=False):
    ''' Coverts a market uid to the datetime string '''
    uidx = [i for i, char in enumerate(uid) if char == '2' and i >= 5][0]
    t0 = uid[uidx:]
    if to_datetime:
        t0 = datetime.datetime.strptime(t0, '%Y%m%d%H%M')
    if return_mkt_spec:
        return t0, uid[:uidx]
    else:
        return t0

def dict_to_json(data_dict, directory, filename):
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Write the data dictionary to a JSON file
    json_file = os.path.join(directory, f'{filename}.json')
    with open(json_file, "w") as f:
        json.dump(data_dict, f, cls=GamsTupleEncoder, indent=4)

    # logging.INFO(f"Successfully dumped GDX data to {json_file}")

def gdx_to_json(directory=None, filename=None, param_names=None, make_time_list=True, 
                return_json=False, save_results=False):
    if directory is None:
        directory = '../system_data/'
    if filename is None:
        filename = 'case5'
    ext = ''
    if filename.split('.')[-1] != 'gdx':
        ext = '.gdx'
    gdx_file_path = os.path.join(directory,filename+ext)
    # Create a GAMS workspace and load GDX
    ws = gams.GamsWorkspace(working_directory=os.getcwd(), debug=gams.DebugLevel.Off)
    db = ws.add_database_from_gdx(gdx_file_path)

    if param_names == None:
        try:
            # Convert the database to a Python dictionary
            data_dict = {}
            # Sets
            data_dict['bus'] = [rec.keys[0] for rec in db['bus']]
            data_dict['circuit'] = [rec.keys[0] for rec in db['circuit']]
            data_dict['line'] = [tuple(rec.keys) for rec in db['line']]
            data_dict['linestatus'] = [tuple(rec.keys) for rec in db['linestatus']]
            data_dict['monitored'] = [tuple(rec.keys) for rec in db['monitored']]
            # Parameters
            data_dict['x'] = encode_tuple(db['x'])
            data_dict['tmax'] = encode_tuple(db['tmax'])
            data_dict['type'] = {rec.keys[0]: rec.value for rec in db['type']}
            data_dict['area'] = {rec.keys[0]: rec.value for rec in db['area']}
            data_dict['baseMVA'] = db['baseMVA'].first_record().value
            
            if return_json:
                return data_dict
            else:
                dict_to_json(data_dict, directory, filename)
    
        except gams.GamsException as e:
            print(f"Error: {e}")
    else:
        json_dict = {}
        time_list = []
        for param in param_names:
            key_list = [tuple(rec.keys) for rec in db[param]]
            val_list = [rec.value for rec in db[param]]
            if 'fwd' in param and make_time_list:
                for key in key_list:
                    time_list += key[1]
            json_dict[param] = {"keys": key_list, "values":val_list}
        if ext == 'gdx':
            filename = filename.split('.')[0:-1]
        if return_json:
            return json_dict
        else:
            dict_to_json(json_dict, directory, filename)
        if make_time_list:
            time_list = list(set(time_list)) # Pull only unique time values
            return time_list

def json_to_gdx(directory = '../system_data/', filename = 'dummy_resource'):
    json_file_path = os.path.join(directory, f'{filename}.json')
    gdx_file_path = os.path.join(directory, f'{filename}.gdx')

    with open(json_file_path, 'r') as f:
        system_data = json.load(f)

    ws = gams.GamsWorkspace(working_directory=os.getcwd(), debug=gams.DebugLevel.Off)
    do = ws.add_database()

    def get_dim_from_list(lst):
        if isinstance(lst, list):
            return len(lst)
        else:
            return 1

    def get_dim_from_dict(dct):
        if 'dim' in dct.keys():
            return dct['dim']
        elif '__tuple__' in dct.keys():
            return len(dct['keys'][0])
        else:
            return 1

    for sym, sym_data in system_data.items():
        if type(sym_data) is float:
            dim = 0
        elif type(sym_data) is list:
            if len(sym_data) == 0 and (sym == 'physical' or sym == 'forward' or sym == 'advisory'):
                dim = 1
            else:
                dim = get_dim_from_list(sym_data[0])
        elif type(sym_data) is tuple:
            dim = len(sym_data[0])
        elif type(sym_data) is dict:
            dim = get_dim_from_dict(sym_data)
        else:
            print('interesting...')
        if sym in gams_symbols['set']:
            gms_set = do.add_set(identifier=sym, dimension=dim)
            for rec in sym_data:
                gms_set.add_record(rec)
        elif sym in gams_symbols['par']:
            gms_par = do.add_parameter(identifier=sym, dimension=dim)
            if type(sym_data) is dict:
                if 'keys' in sym_data.keys():
                    try:
                        parsed_sym = dict(zip(sym_data['keys'], sym_data['values']))
                    except TypeError:
                        parsed_keys = [tuple(k) for k in sym_data['keys']]
                        parsed_sym = dict(zip(parsed_keys, sym_data['values']))
                else:
                    parsed_sym = sym_data
                for key, value in parsed_sym.items():
                    gms_par.add_record(key).value = value
            elif type(sym_data) is float:
                gms_par.add_record().value = sym_data
            elif '__tuple__' in sym_data.keys():
                for key, value in dict(zip(sym_data['keys'], sym_data['values'])).items():
                    gms_par.add_record(keys=key).value = value
            else:
                print(sym_data)
        else:
            raise ValueError(f"Handling of symbol {sym} not specified.")
    do.export(gdx_file_path)
    logging.info(f"Market data saved to {gdx_file_path}.")

def make_from_dict(dirs, run_dir):
    '''Helper function for make_market_dirs to help with multi-level recursion'''
    for td, ldirs in dirs.items():
        os.makedirs(join(run_dir, td), exist_ok=True)
        if ldirs is not None:
            if type(ldirs) == dict:
                make_from_dict(ldirs, join(run_dir, td))
            else:
                for ld in ldirs:
                    dirpath = join(run_dir, td, ld)
                    os.makedirs(dirpath, exist_ok=True)

def make_market_dirs(run_dir):
    ''' Makes all of the directories/subdirectories needed for this market simulation '''
    dirs = {'market_clearing':['system_data','offer_data', 'results'],
            'physical_dispatch':{'storage':['data', 'results']}, 'settlement':None}
    make_from_dict(dirs, run_dir)

def add_gams_parameter(database:gams.GamsDatabase, par_name, domain):
    try:
        gms_par = database.get_parameter(parameter_identifier=par_name)
    except:
        gms_par = database.add_parameter_dc(identifier=par_name, domains=domain)
    return gms_par

def add_gams_set(database:gams.GamsDatabase, record, set_name='resource', set_dimension=1):
    try:
        gms_set = database.get_set(set_identifier=set_name)
    except:
        gms_set = database.add_set(identifier=set_name, dimension=set_dimension)
    if record not in gms_set:
        gms_set.add_record(record)

def offer_data_parser(offer_dict):
    if 'domain' not in offer_dict.keys():
        raise KeyError(f"Domain expected in offer dictionary but {offer_dict.keys()} keys received.")
    elif len(offer_dict['domain']) == 1:
        parsed_offer = {None: offer_dict['values']}
    elif len(offer_dict['domain']) == 2:
        parsed_offer = dict(zip(offer_dict['keys'], offer_dict['values']))
    elif len(offer_dict['domain']) >= 3:
        parsed_keys = [tuple(k) for k in offer_dict['keys']]
        parsed_offer = dict(zip(parsed_keys, offer_dict['values']))
    else:
        raise ValueError(f"Domain size should be positive integer but is {len(offer_dict['domain'])}.")
    return parsed_offer

def make_zero_offer(rid, times, resources_df):
    """ Creates an offer with all marginal cost bids set to zero (single blocks) """
    str_df = resources_df['Storage']
    attributes = str_df.loc[str_df['rid'] == rid]
    offer = {'bid_soc': False}
    for key, domain in parameter_domains.items():
        if key in attributes:
            attr_val = attributes[key].values[0]
            if 'offer_block' in domain:
                if 'mq' in key:
                    if 'ch' in key:
                        attr_val = attributes['chmax'].values[0]
                    elif 'dc' in key:
                        attr_val = attributes['dcmax'].values[0]
                    elif 'soc' in key:
                        attr_val = attributes['socmax'].values[0]
                elif 'mc' in key:
                    attr_val = 0
            if 't' in domain:
                key_dict = {}
                for t in times:
                    key_dict[t] = float(attr_val)
                offer[key] = key_dict
            else:
                offer[key] = float(attr_val)
    zero_offer = {rid: offer}
    return zero_offer

def clear_all_offers(pid='all', run_dir='.'):
    ''' Clears all offers in participant directories and in previous saves directory. '''
    offer_dir = join(run_dir, 'market_clearing/offer_data')
    # First clear participant directories
    part_dirs = [d for d in glob.glob(join(offer_dir, 'participant_*')) if pid in d or pid=='all']
    for part_dir in part_dirs:
        offers = [f for f in glob.glob(join(part_dir, 'offer_*.json'))]
        for offer in offers:
            os.remove(offer)
    # Next clear previous offer directory
    prev_offers = [f for f in glob.glob(join(offer_dir, 'previous_offers', 'prev_*.json'))]
    for prev_offer in prev_offers:
        os.remove(prev_offer)

def save_previous_offer(offer, rid, uid, run_dir='.'):
    ''' Saves the offer by rid and mkt_type. Overwrites any previous saved offer. '''
    tjunk, mkt_type = uid_to_time(uid, return_mkt_spec=True)
    prev_offer_dir = join(run_dir, 'market_clearing/offer_data/previous_offers')
    if not os.path.isdir(prev_offer_dir):
        os.makedirs(prev_offer_dir)
    offer_out = {rid: offer}
    with open(join(prev_offer_dir, f'prev_{rid}_{mkt_type}.json'), 'w') as f:
        json.dump(offer_out, f, indent=4, cls=fl.NpEncoder)
        
def load_previous_offer(rid, uid, run_dir='.'):
    ''' Loads the previous offer from its json file '''
    tjunk, mkt_type = uid_to_time(uid, return_mkt_spec=True)
    prev_offer_dir = join(run_dir, 'market_clearing/offer_data/previous_offers')
    try:
        with open(join(prev_offer_dir, f'prev_{rid}_{mkt_type}.json'), 'r') as f:
            prev_offer = json.load(f)
        return prev_offer
    except:
        return None

def fill_from_previous_offer(rid, uid, prev_offer=None, run_dir='.'):
    ''' Looks for a previous offer with matching mkt_type and rid. If found, it will update
        the timestamps and return the new offer.
        If not found it will return None.
    '''
    print(f"Filling from previous offer for uid {uid}")
    if prev_offer is None:
        prev_offer = load_previous_offer(rid, uid, run_dir=run_dir)
        if prev_offer is None:
            return None
    # Verify that prev_offer is indexed by rid
    if list(prev_offer.keys())[0] != rid:
        logger = logging.getLogger()
        logger.debug(f"Previous offer for {rid} at interval {uid} does not have correct key structure.")
        return None
    times = get_time_from_system(uid, run_dir=run_dir)
    new_offer = {}
    for key, value in prev_offer[rid].items():
        # All time series options will have a dictionary as the value
        # TODO: make update for initial energy (use previous dispatched energy for RTM. Not sure for DAM...)
        if type(value) == dict:
            new_val_dict = {}
            for i, quantity in enumerate(value.values()):
                new_val_dict[times[i]] = quantity
            new_offer[key] = new_val_dict
        else:
            new_offer[key] = value
    filled_offer = {rid: new_offer}
    return filled_offer

def prep_offer_for_gdx(pid, offer_dict, resources_df, participant_res, times=None):
    ''' Converts simplified competitor offer format to gdx compatible offer format '''
    prepped_offer = {}
    for rid in offer_dict.keys():
        if rid not in participant_res[pid] and 'p' != rid[0]:
            raise ValueError(f'Found invalid rid {rid} in submission by {pid}')
        # First update the 'attributes' key
        if 'p' == rid[0]:
            rtype = 'vir'
            rbus = rid.split('_')[-1]
        else:
            rtype = resources_df['Master'].query(f"rid=='{rid}'")['ResourceType'].values[0]
            rbus = resources_df['Master'].query(f"rid=='{rid}'")['ResourceBus'].values[0]
            bid_soc = offer_dict[rid]['bid_soc']
        attr = {'resource_name':rid, 'participant_name':pid, 'bus_location':rbus,
                'resource_type':rtype}
        if 'attributes' in prepped_offer.keys():
            prepped_offer['attributes'].update({f'{rtype}_{rid}':attr})
        else:
            prepped_offer['attributes'] = {f'{rtype}_{rid}':attr}
        # Next update the 'offer' key
        inner_dict = {}
        for key, value in offer_dict[rid].items():
            if key == 'bid_soc':
                continue
            value_dict = {}
            value_dict['domain'] = parameter_domains[key]
            if 'offer_block' in parameter_domains[key] and 't' in parameter_domains[key]:
                # Logic to zero soc or ch/dc mc curves based on bid_soc_value
                mult = 1
                if ('ch' in key or 'dc' in key or 'soc' in key) and 'mc' in key:
                    if bid_soc and ('ch' in key or 'dc' in key):
                        mult = 0
                    elif not bid_soc and 'soc' in key:
                        mult = 0
                tlist = [t for t in value.keys()]
                klist, vlist = [], []
                block_t = list(value.values())
                for i, t in enumerate(tlist):
                    if type(block_t[i]) != list:
                        klist += [['1', t]]
                        vlist += [block_t[i]*mult]
                    else:
                        for nb, b in enumerate(block_t[i]):
                            klist += [[str(nb+1), t]]
                            vlist += [b*mult]
                value_dict['keys'] = klist
                value_dict['values'] = vlist
            elif 't' in parameter_domains[key]:
                klist = [t for t in value.keys()]
                vlist = [v for v in value.values()]
                value_dict['keys'] = klist
                value_dict['values'] = vlist
            else:
                value_dict['values'] = value
            inner_dict[key] = value_dict
        if 'offer' in prepped_offer.keys():
            prepped_offer['offer'].update({f'{rtype}_{rid}':inner_dict})
        else:
            prepped_offer['offer'] = {f'{rtype}_{rid}':inner_dict}
    return prepped_offer

def offer_to_gdx(data_dict, filename, directory='../offer_data/'):
    location = directory + filename
    gdxfile = location + '.gdx'

    ws = gams.GamsWorkspace(working_directory=os.getcwd(), debug=gams.DebugLevel.Off)
    do = ws.add_database()
    
    for resource_name, offers in data_dict['offer'].items():
        add_gams_set(do, resource_name, set_name='resource', set_dimension=1)
        rtype = data_dict['attributes'][resource_name]['resource_type']
        add_gams_set(do, resource_name, set_name=f"{rtype}", set_dimension=1)
        found_soc, found_ch_dc = False, False
        # Add block sets for gams (dem_block(n,t), gen_block(n,t), etc)
        for key in offers.keys():
            if 'block' and 'mq' in key:
                # Drawing nblocks from the first element of the last keys entry
                nblocks = 1
                for kval in offers[key]['keys']:
                    nblocks = max(nblocks, int(kval[0]))
                extra = ''
                if '_soc_' in key:
                    extra = '_soc'
                    found_soc = True
                elif '_dc' in key or '_ch' in key:
                    if found_ch_dc:
                        continue
                    extra = '_en'
                    found_ch_dc = True
                    # Check both charge/discharge and keep the highest nblocks
                    for kval in offers['block_dc_mq']['keys']:
                        nblocks = max(nblocks, int(kval[0]))
                    for kval in offers['block_ch_mq']['keys']:
                        nblocks = max(nblocks, int(kval[0]))
                for block in range(1,nblocks+1):
                    add_gams_set(do, [resource_name, str(block)], set_name=f"{rtype}{extra}_block", set_dimension=2)
                if rtype == 'str':
                    if found_soc and found_ch_dc:
                        break
                else:
                    break
        # Add resource_at_bus sets for gams
        bus = str(data_dict['attributes'][resource_name]['bus_location'])
        add_gams_set(do, [resource_name, bus], set_name='resource_at_bus', set_dimension=2)
        for par_name, offer_data in offers.items():
            if not offer_data:
                do.add_parameter(par_name, dimension=0) # TODO: make sure empty (null) offer parameters get passed to gdx with correct domains
                continue
            elif not 'domain' in offer_data.keys():
                raise LookupError(f"Expected 'domain' in offer_data keys. Found {offer_data.keys()}.")
            gms_par = add_gams_parameter(do, par_name, domain=offer_data['domain'])
            parsed_offer = offer_data_parser(offer_data)
            for key, value in parsed_offer.items():
                rec_key = [resource_name]
                if key is None:
                    pass
                elif type(key) is tuple:
                    rec_key += [k for k in key]
                else:
                    rec_key.append(key)
                gms_par.add_record(rec_key).value = value
    do.export(gdxfile)
    logging.info(f"Offer data saved to {gdxfile}.")

def handle_TH(bas):
    ''' Helper function to handle the TH_{area} balancing authority name (when splitting on '_' )
        Returns a length 2 list with TH in the right spot (there are no TH - TH lines in 
                                                           our WECC system right now)
    '''
    if len(bas) == 3:
        th_idx = bas.index('TH')
        if th_idx == 0:
            bas = [f'{bas[0]}_{bas[1]}', bas[2]]
        elif th_idx == 1:
            bas = [bas[0], f'{bas[1]}_{bas[2]}']
    elif len(bas) > 3:
        raise ValueError(f'Do not know how to interpret a line name using 3 _ ({bas})')
    return bas

def parse_network_data(network_df):
    """ Takes the transmission line and BA info and converts to a network dictionary """

    lines = network_df['transmission_line'].values.flatten()
    tmaxes = network_df['max_mw'].values.flatten()
    ba_list = []
    line_list = []
    tmax_list = []
    # Go through to identify a unique line list and BA list
    duplicate_inds = []
    for cnt, line in enumerate(lines):
        bas = line.split('_')
        # Check for any of the TH_* BAs - if so, join back into a length two list
        bas = handle_TH(bas)
        # Checking for lines with same terminal BA, regardless of origin
        for cnt2, found_line in enumerate(line_list):
            if set(bas) == set(found_line[:2]):
                duplicate_inds += [[cnt, cnt2]]
        # Last element of the below is circuit - may need to increment if duplicates exist
        bas_circuit = [bas[0], bas[1], '1']
        # If a duplicate was found, do not add it to the line list, but add capacity to other line
        if len(duplicate_inds) == 0:
            line_list += [bas_circuit]
            tmax_list += float(tmaxes[cnt]),
        elif cnt not in duplicate_inds[-1]:
            line_list += [bas_circuit]
            tmax_list += float(tmaxes[cnt]),
        else:
            tmax_list[duplicate_inds[-1][1]] += float(tmaxes[cnt])
        # Make a running list of all balancing authorities
        for ba in bas:
            if ba not in ba_list:
                ba_list += ba,
    # Now fill out sub-dictionaries with proper info
    # Read in the selected topology configuration
    with open("topology_conf.json", "r") as f:
        topology = json.load(f)
    scaling = topology["scaling"]
    circuit = ["1"]
    x_dict = {"__tuple__": 'true'}
    x_dict['keys'] = line_list
    x_dict['values'] = [1/(tm/scaling) for tm in tmax_list]
    tmax = {"__tuple__": 'true'}
    tmax['keys'] = line_list
    tmax['values'] = [tm/scaling for tm in tmax_list]
    bus_type = {}
    areas = {}
    for cnt, ba in enumerate(ba_list):
        areas[ba] = 1
        # Just have a reference (1) bus and load (2) buses - can add slack (3) if needed
        if cnt == 0:
            bus_type[ba] = 1
        else:
            bus_type[ba] = 2
    # Finally, fill all of the keys into the overall network dictionary
    network_dict = {}
    network_dict['bus'] = ba_list
    network_dict['circuit'] = circuit
    network_dict['line'] = line_list
    network_dict['x'] = x_dict
    network_dict['tmax'] = tmax
    network_dict['type'] = bus_type
    network_dict['area'] = areas
    network_dict['baseMVA'] = scaling
    return network_dict

def load_participant_info(part_dir=None):
    """ Loads participant info from an Excel file into a directory """
    if part_dir is None:
        part_dir = 'market_clearing/offer_data/'
    # df_participant = pd.read_excel(os.path.join(part_dir, 'participant_info.xlsx'))
    # pids = list(df_participant.pid)
    # participant_res = {}
    # for idx, pid in enumerate(pids):
    #     participant_res[pid] = df_participant.ResourceList[idx].split(',')
    participant_res = {}
    with open(os.path.join(part_dir, 'participant_info.json'), 'r') as f:
        part_info = json.load(f)
    for username, pdict in part_info.items():
        pid, rlist = list(pdict.items())[0]
        if pid in participant_res.keys():
            raise KeyError(f'Duplicate participant id {pid} found in participant_info.json')
        participant_res[pid] = rlist
    return participant_res

def load_resource_info(res_dir=None):
    """ Loads resource info from an Excel file into a dict of pandas dataframes """
    if res_dir is None:
        res_dir = 'market_clearing/offer_data/'
    resources_df = pd.read_excel(os.path.join(res_dir, 'resource_info.xlsx'),sheet_name=None)
    return resources_df

def get_time_from_system(uid, run_dir='.'):
    # Import system data, most relevantly timeseries (system_data['t'])
    system_dir = join(run_dir,'market_clearing/system_data')
    json_file_path = os.path.join(system_dir, f'{uid}.json')
    with open(json_file_path, 'r') as f:
        system_data = json.load(f)
    return system_data['t']

def drw_mr(prev, sig=0.3, mr=0.1):
    ''' Damped random walk with a mean reversion term'''
    drw = prev + np.random.randn()*sig - mr*(prev**2)*np.sign(prev)
    return drw

def interp_forecast(forecast, actual, durations, drw):
    ''' Modifies the forecast by applying a logarithmic interpolation
        ranging from 0 (current time) to 1 (end of durations).
        Also applies a damped random walk multiplier
    '''
    tscale = np.ones(len(durations))
    for i in range(len(durations)-1):
        tscale[i+1] = tscale[i] + durations[i]
    interp = np.log(tscale)
    interp /= np.max(interp)
    error = np.array(forecast) - np.array(actual)
    # print("DRW:", drw)
    # plt.plot(interp)
    # plt.plot(interp*drw)
    # plt.show()
    # plt.plot(error)
    # print(interp, drw)
    error *= interp*drw
    # plt.plot(error)
    # print(np.mean(forecast), " : ", np.mean(error), " : ",  np.mean(np.array(actual) + error))
    # plt.show()
    forecast_out = np.array(actual) + error
    forecast_out[forecast_out < 0] = 0
    return list(forecast_out)

def open_forecast_file(rtype, ttype, actual=False, ftype='parquet', run_dir='.'):
    ''' Opens a forecast for a resource and time type (ttype = '5min' or '1hr') 
        actual: if True uses actual instead of forecast
        ftype: can be parquet or csv
    '''
    system_dir = join(run_dir,'market_clearing/system_data')
    # Read in the necessary files
    if ftype == 'parquet':
        pfunc = pd.read_parquet
    elif ftype == 'csv':
        pfunc = pd.read_csv
    if rtype == 'hydro': # Not using a forecast for hydro
        df = pfunc(join(system_dir,f'{rtype}_{ttype}.{ftype}'))
    elif actual:
        df = pfunc(join(system_dir,f'{rtype}_{ttype}_actual.{ftype}'))
    else:
        df = pfunc(join(system_dir,f'{rtype}_{ttype}_forecast.{ftype}'))
    # Drop extra balancying authorities
    with open(join(run_dir, "ba_names.json"), "r") as f:
        use_ba_dict = json.load(f)
    use_bas = sorted(use_ba_dict["ba"])
    for df_ba in df.columns:
        if df_ba not in use_bas:
            df = df.drop(df_ba, axis=1)
    return df

def merge_forecast_files(rlist, lookup, mode='ba', ftype='parquet', actual=False, run_dir='.'):
    ''' Loads the renewable and demand values from the forecast spreadsheet 
        at the given timestamps.'''
    # Read in the scale factors (computed in load_gen_info.py)
    with open(join(run_dir,"scale_factors.json"), "r") as f:
        scale_factors = json.load(f)
    scale = {'demand':scale_factors['flex_dem'], 'wind':scale_factors['wind'],
             'solar':scale_factors['solar'], 'hydro':scale_factors['other_ren']}
    forecasts = {}
    durations = lookup['duration']
    num_avg = lookup['num_avg']
    file_idx = lookup['file_idx']
    fdurations = {'5min':(5 in durations), '1hr':(60 in durations)}
    for rtype in rlist:
        for key, tf in fdurations.items():
            if tf:
                full_df = open_forecast_file(rtype, key, ftype=ftype, actual=actual)
                if mode == 'total':
                    total = full_df.sum(axis=1)*scale[rtype]
                    total[total<0] = 0 # No negative values
                    forecasts[f'{rtype}_{key}'] = total
                elif mode == 'ba':
                    for ba in full_df.columns:
                        ba_vals = full_df[ba].values*scale[rtype]
                        ba_vals[ba_vals<0] = 0 # No negative values
                        try:
                            forecasts[f'{rtype}_{key}'].update({ba: ba_vals})
                        except KeyError:
                            forecasts[f'{rtype}_{key}'] = {ba: ba_vals}
    power = {}
    if mode == 'total':
        for rtype in rlist:
            power[rtype] = []
            for i in range(len(durations)):
                fi = file_idx[i]
                if durations[i] == 5:
                    if num_avg[i] > 1:
                        value = np.mean(forecasts[f'{rtype}_5min'].iloc[fi:fi+num_avg[i]])
                    else:
                        value = forecasts[f'{rtype}_5min'].iloc[fi]
                elif durations[i] == 60:
                    value = forecasts[f'{rtype}_1hr'].iloc[fi]
                power[rtype] += [value]
    elif mode == 'ba':
        for rtype in rlist:
            ba_dict = {}
            try:
                ba_list = [ba for ba in forecasts[f'{rtype}_5min'].keys()]
            except KeyError:
                ba_list = [ba for ba in forecasts[f'{rtype}_1hr'].keys()]
            for ba in ba_list:
                ba_dict[ba] = []
                for i in range(len(durations)):
                    fi = file_idx[i]
                    if durations[i] == 5:
                        if num_avg[i] > 1:
                            value = np.mean(forecasts[f'{rtype}_5min'][ba][fi:fi+num_avg[i]])
                        else:
                            value = forecasts[f'{rtype}_5min'][ba][fi]
                    elif durations[i] == 60:
                        value = forecasts[f'{rtype}_1hr'][ba][fi]
                    ba_dict[ba] += [value]
            power[rtype] = ba_dict
    return power

def get_forecast(uid, intervals, mode='ba', rlist=['all'], actual=False, times=None,
                 return_intervals=False, hist=False, run_dir='.', drw=1, interp=True):
    ''' Loads the forecast at the appropriate intervals
        Inputs:
            uid: market unique id
            intervals: interval durations from mktSpec
            mode: can be 'ba' or 'total'. If 'total' returns total by type. If 'ba' returns
                  a nested dictionary by balancing authority
            rlist: a list of resources ['demand', 'wind', 'solar', 'hydro'] or ['all']
            return_intervals: if True returns a dictionary of interval durations (used for
                              the market info sent to participants)
            hist: boolean for if using history (ignores interval block info, duration=5 always)
            times: allows you to set a specific set of times at which to get the forecast
            run_dir: can set a different directory for a particular simulation instance
    '''
    if 'all' in rlist:
        rlist = ['demand', 'wind', 'solar', 'hydro']
    if times is None:
        times = get_time_from_system(uid, run_dir=run_dir)
    elif type(times) == str:
        times = [times]
    interval_dict = {}
    forecast_lookup = {'interval_idx':[], 'duration':[], 'num_avg':[], 'file_idx':[]}
    year = times[0][0:4]
    t0 = datetime.datetime.strptime(f'{year}01010000', '%Y%m%d%H%M')
    int_block = 0
    extra = 0
    for i, t in enumerate(times):
        # Forecast Lookup
        forecast_lookup['interval_idx'] += [i]
        if hist:
            duration = 5
        else:
            duration = intervals[int_block][1]
        if duration == 5 or duration == 60:
            num_avg = 1
        else:
            num_avg = int(duration/5)
            duration = 5
        forecast_lookup['duration'] += [duration]
        forecast_lookup['num_avg'] += [num_avg]
        # Time step converted to file index
        dt = (datetime.datetime.strptime(t, '%Y%m%d%H%M')-t0).total_seconds()
        if duration == 5:
            dt /= 60 # Convert to minutes
            file_idx = int(dt/5)
        elif duration == 60:
            dt /= 3600 # Convert to hours
            file_idx = int(dt)
        forecast_lookup['file_idx'] += [file_idx]
        # Interval Dictionary
        if not hist:
            interval_dict[t] = intervals[int_block][1]
            if i+1 == intervals[int_block][0] + extra:
                extra += intervals[int_block][0]
                int_block += 1
    forecast = merge_forecast_files(rlist, forecast_lookup, mode=mode, actual=actual)
    if interp:
        true_vals = merge_forecast_files(rlist, forecast_lookup, mode=mode, actual=True)
        for rtype, val in forecast.items():
            if type(val) == dict:
                # print(forecast['wind']['AVA'], true_vals['wind']['AVA'])
                for ba in val.keys():
                    forecast[rtype][ba] = interp_forecast(forecast[rtype][ba], true_vals[rtype][ba],
                                                  forecast_lookup['duration'], drw)
            else:
                forecast[rtype] = interp_forecast(forecast[rtype], true_vals[rtype],
                                              forecast_lookup['duration'], drw)
    if return_intervals:
        return forecast, interval_dict
    else:
        return forecast

def get_actual(uid, intervals, mode='ba', rlist=['demand', 'solar', 'wind'], times=None, hist=False,
               run_dir='.'):
    ''' Gets the actual values for resources in rlist for given mode (ba or total)
        (this wraps around get_forecast with actual=True)
    '''
    actual = get_forecast(uid, intervals, mode=mode, rlist=rlist, actual=True, hist=hist, 
                          times=times, run_dir=run_dir, interp=False)
    return actual

def join_actual_forecast(actual, forecast):
    ''' Adds the actual value to the first part of the forecast (for renewables/demand 
        cleared on physical intervals)
    '''
    forecast_out = {}
    for rtype, val1 in actual.items():
        if type(val1) == dict:
            rtype_dict = {}
            for ba in val1.keys():
                rtype_dict[ba] = actual[rtype][ba] + forecast[rtype][ba]
            forecast_out[rtype] = rtype_dict
        else:
            forecast_out[rtype] = actual[rtype] + forecast[rtype]
    return forecast_out

def get_mkt_for_participants(resources_df, participant_res, uid, prev_uid, intervals, run_dir='.',
                             save=True, drw=1):
    """ Returns (or saves as .json) a dictionary with the following:
        'uid':uid
        'intervals':{timestamp:duration}
        'forecast':{'wind':[],'solar':[],'demand':[]}
    """
    mkt_out = {'uid':uid}
    time, mkt_spec = uid_to_time(uid, return_mkt_spec=True)
    time_limits = {'TSDAM':360, 'TSRTM':5, 'MSDAM':360, 'MSRTM':5, 'RHF36':15, 'RHF12a':7,
                   'RHF12b':7, 'RHF12c':7, 'RHF2a':5, 'RHF2b':5}
    mkt_out['time_limit'] = time_limits[mkt_spec]
    offer_dir=join(run_dir,'market_clearing/offer_data/')
    forecast, interval_dict = get_forecast(uid, intervals, mode='total', 
                                           rlist=['demand', 'solar', 'wind'],
                                           return_intervals=True, run_dir=run_dir, drw=drw)
    mkt_out['intervals'] = interval_dict
    forecast_out = {}
    rlist = ['wind', 'solar', 'demand']
    for rtype in rlist:
        forecast_trunc = truncate_float(forecast[rtype])
        if rtype == 'demand':
            rtype = 'load' # Rename output for participants
        forecast_out[rtype] = forecast_trunc
    mkt_out['forecast'] = forecast_out
    if save:
        with open(os.path.join(offer_dir,'forecast.json'), "w") as f:
            json.dump(forecast_out, f, indent=4, cls=fl.NpEncoder)
    history = get_history(prev_uid, (prev_uid == None), intervals, run_dir=run_dir)
    mkt_out['history'] = history
    return mkt_out
    
def get_history(prev_uid, newrun, intervals, max_hours=24*365, run_dir='.', savedir='records'):
    ''' Creates a json of the actual wind/solar/demand and LMP to date.
        Format is:
        'history':{prices:{'en':[],'rgu':[],...,'nsp':[]}
                   'times':[],'wind':[],'solar':[],'demand':[]} (actuals)
        Also saves the latest history updates to csv
    '''
    # For now set a local history limit (cluster won't have a problem with the larger files)
    if run_dir == '.':
        max_hours = 24
    offer_dir = join(run_dir, 'market_clearing/offer_data')
    results_dir = join(run_dir, 'market_clearing/results')
    savedir = join(run_dir, savedir)
    if newrun:
        if os.path.isfile(os.path.join(offer_dir,'history.json')):
            os.remove(os.path.join(offer_dir,'history.json'))
        return None
    pdict = {'EN':[], 'RGU':[], 'RGD':[], 'SPR':[], 'NSP':[]}
    hist_out = {'times':[],'wind': [], 'solar':[], 'load':[], 'prices':pdict}
    # First get the actuals from the forecast for the last timestep
    times = get_time_from_system(prev_uid, run_dir=run_dir)
    this_time = [times[0]]
    if os.path.isfile(os.path.join(offer_dir,'history.json')):
        with open(os.path.join(offer_dir,'history.json'), 'r') as f:
            hist_out = json.load(f)
    this_datetime = datetime.datetime.strptime(times[0], '%Y%m%d%H%M')
    if len(hist_out['times']) > 0:
        first_hist_time = datetime.datetime.strptime(hist_out['times'][0], '%Y%m%d%H%M')
    else:
        first_hist_time = this_datetime
    delta_hours = (this_datetime-first_hist_time).total_seconds()/3600.
    remove_hist = False
    if delta_hours > max_hours:
        remove_hist = True
    # First do actual loads
    hist_out['times'] += this_time
    if remove_hist:
        del hist_out['times'][0]
    rlist = ['wind', 'solar', 'demand']
    power = get_actual(prev_uid, intervals, mode='total', rlist=['demand', 'solar', 'wind'],
                       times=this_time, hist=True, run_dir='.')
    for rtype in rlist:
        power_trunc = truncate_float(power[rtype])
        if rtype == 'demand':
            rtype = 'load' # Rename output for participants
        hist_out[rtype] += power_trunc
        if remove_hist:
            del hist_out[rtype][0]
    # Now get updated average LMP
    ws = gams.GamsWorkspace(working_directory=os.getcwd())
    mkt_results = ws.add_database_from_gdx(os.path.join(results_dir,f'results_{prev_uid}.gdx'))
    multiplier = 12 # for five minute intervals only (which these all should be...)
    lw_lmp = [float(rec.value*multiplier) for rec in mkt_results['lw_lmp'] if rec.keys[0] == this_time[0]]
    if len(lw_lmp) == 0:
        lw_lmp == [0]
    lw_lmp = truncate_float(lw_lmp)
    hist_out['prices']['EN'] += lw_lmp
    # Add in regulation/reserves
    reg_types = ['RGU', 'RGD', 'SPR', 'NSP']
    for reg in reg_types:
        reg_val = [float(rec.value*multiplier) for rec in mkt_results['mcp'] if \
                   (rec.keys[0]==this_time[0]) and (rec.keys[1]==reg)]
        if len(reg_val) == 0:
            reg_val = [0]
        reg_val = truncate_float(reg_val)
        hist_out['prices'][reg] += reg_val
    if remove_hist:
        for ctype in hist_out['prices'].keys():
            del hist_out['prices'][ctype][0]
    with open(os.path.join(offer_dir,'history.json'), "w") as f:
        json.dump(hist_out, f, indent=4, cls=fl.NpEncoder)
    return hist_out

def truncate_float(value, decimals=6):
    ''' truncates a float to the given number of decimals (rounding the last position) '''
    if type(value) == float:
        value = round(value, decimals)
    elif type(value) == list or type(value) == np.ndarray:
        # print(value)
        for i in range(len(value)):
            if type(value[i]) == list or type(value[i]) == tuple:
                value[i] = truncate_float(value[i], decimals)
            else:
                value[i] = round(value[i], decimals)
    elif type(value) == tuple:
        new_tuple = ()
        for i in range(len(value)):
            new_value = round(value[i], decimals)
            new_tuple += (new_value,)
    return value
    
def get_res_for_participants(pid, prev_uid, resources_df, participant_res, degradation, run_dir='.', 
                             save=False):
    ''' Loads the latest resource status as well as the ledger/settlement information for the
        given participant (pid).
    '''
    offer_dir = join(run_dir, 'market_clearing/offer_data')

    dispatch_dir = join(run_dir, 'physical_dispatch/storage/results')
    # Load dispatched status
    if prev_uid != None:
        ws = gams.GamsWorkspace(working_directory=os.getcwd())
        gams_db = ws.add_database_from_gdx(os.path.join(dispatch_dir,f'dispatch_{prev_uid}.gdx'))
        soc_records = gams_db["soc"]
        temp_records = gams_db["temp"]
        dispatch_records = gams_db["dispatch"]
    # Loop through all participants and write a status file for each
    return_dict = {}
    for p, rlist in participant_res.items():
        if pid != p:
            continue
        # Load status into a dictionary for writing to json file
        status_dict = {}
        all_resources = {}
        for rid in rlist:
            if prev_uid == None:
                soc = resources_df['Storage'].query(f"rid=='{rid}'")['soc_begin'].values[0]
                dispatch = resources_df['Storage'].query(f"rid=='{rid}'")['init_en'].values[0]
                temp = 20
            else:
                soc = soc_records.find_record(rid).value
                temp = temp_records.find_record(rid).value
                try:
                    dispatch = dispatch_records.find_record(rid).value
                except gams.control.workspace.GamsException:
                    dispatch = 0
            degradation_trunc = copy.copy(degradation[rid])
            for k, v in degradation_trunc.items():
                degradation_trunc[k] = truncate_float(v)
            resource_dict = {'soc': float(soc), 'temp': float(temp), 'dispatch':float(dispatch),
                             'degradation':degradation_trunc}
            all_resources[f'{rid}'] = resource_dict
        status_dict['status'] = all_resources
        if save:
            pdir = os.path.join(offer_dir,f'participant_{pid}')
            if not os.path.exists(pdir):
                os.makedirs(pdir)
            part_file = os.path.join(pdir,'status.json')
            with open(part_file, "w") as f:
                json.dump(status_dict, f, indent=4, cls=fl.NpEncoder)
        return_dict[p] = status_dict
    return return_dict

def convert_ledger_to_score(uid, ledger, settlement, degradation):
    ''' Uses the ledger object to show the scheduled power (for future only, no history)
        Takes the settlement and degradation costs and makes a subtotal (scores) dictionary 
    '''
    this_time = uid_to_time(uid, to_datetime=True)
    # Go through and sum the scheduled quantity on ledger timestamps only if they are >= this_time
    schedule = {}
    for resource, products in ledger.items():
        for cost_type, value_dict in products.items():
            for tstamp, qplist in value_dict.items():
                t = f'{tstamp[:4]}-{tstamp[4:6]}-{tstamp[6:8]}T{tstamp[8:10]}:{tstamp[10:12]}:00'
                date_tstamp = datetime.datetime.fromisoformat(t)
                if date_tstamp >= this_time:
                    if resource not in schedule.keys():
                        schedule[resource] = {}
                    if cost_type not in schedule[resource].keys():
                        schedule[resource][cost_type] = {}
                    if tstamp not in schedule[resource][cost_type].keys():
                        schedule[resource][cost_type][tstamp] = 0
                    for qp in qplist:
                        schedule[resource][cost_type][tstamp] += qp[0]*1.0
    scores = {"net_revenue": {}, "degradation_cost": {}, "profit": {}, "latest": 0}
    # Include degradation cost (flip sign, input is negative, output is positive "cost")
    for rid, deg_dict in degradation.items():
        for tstamp, income in deg_dict.items():
            if tstamp in scores["degradation_cost"].keys():
                scores["degradation_cost"][tstamp] -= 1.0*income
            else:
                scores["degradation_cost"][tstamp] = -1.0*income
    # Sum up the settlement over all products at each timestamp
    net_revenue = {}
    for resource, products in settlement.items():
        for cost_type, value_dict in products.items():
            for tstamp, income in value_dict.items():
                if tstamp in net_revenue.keys():
                    net_revenue[tstamp] += 1.0*income
                else:
                    net_revenue[tstamp] = 1.0*income
    scores["net_revenue"] = net_revenue
    # Now sum up the profits by timestamp and total
    for tstamp, income in net_revenue.items():
        if tstamp in scores["degradation_cost"].keys():
            income -= scores["degradation_cost"][tstamp]
        scores["profit"][tstamp] = 1.0*income
        scores["latest"] += 1.0*income
    return schedule, scores

def append_csv(filename, newline, header=None, savedir='saved', run_dir='.'):
    ''' Appends the newline to a csv. If keyword 'header' is specified it will 
        start a new csv file'''
    if not os.path.isdir(join(run_dir,savedir)):
        os.makedirs(join(run_dir,savedir))
    savefile = os.path.join(run_dir, savedir, filename)
    if header is not None:
        with open(savefile, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(header)
    with open(savefile, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(newline)

def update_power_dict(power_dict, record, rid, rtype, rbus, by_ba, transmission=False):
    ''' Helper function to update the dictionary in save_mkt_power depending
        on whether saving by balancing authority or generator and whether saving
        demand or generation
        Note, storage is presently being saved in generation (regardless of 
        whether the unit is charging or discharging). This is to help see how
        much of the demand is being used for flex
    '''
    cname = 'power'
    if rtype == 'dem':
        cname = 'demand'
    if transmission:
        by_ba = False # Currently have no way to do both at once (not meaningful)
    if by_ba:
        if f'{cname}_{rbus}' in power_dict.keys():
            power_dict[f'{cname}_{rbus}'] += record.value
        else:
            power_dict.update({f'{cname}_{rbus}': record.value})
    elif transmission:
        bus1, bus2 = rid, rtype
        power_dict.update({f'{bus1}_{bus2}': record.level})
    else:
        power_dict.update({f'{cname}_{rid}': record.value})
    return power_dict

def save_mkt_pow(uid, rmaster, newrun=False, savedir='saved', run_dir='.', by_ba=True):
    ''' Saves the market clearing results in a physical dispatch period for all 
        system generators and demand as well as the power flow on each of the 
        transmission lines.
        If by_ba = True, will aggregate by balancing authority, otherwise will save
        each unit separately
    '''
    this_time = uid_to_time(uid, to_datetime=True)
    this_time_str = uid_to_time(uid)
    ws = gams.GamsWorkspace(working_directory=os.getcwd())
    mkt_dir = join(run_dir, 'market_clearing/results')
    mkt_results = ws.add_database_from_gdx(os.path.join(mkt_dir,f'results_{uid}.gdx'))
    # First do generation and demand
    power = mkt_results["actual"]
    # Sort through all the records and pull out the storage resource target dispatch levels
    power_dict = {}
    found_rids = []
    for i in range(power.number_records):
        record = power.next()
        rtype, rid = record.keys[0].split('_')
        if 'vir' in rtype: # Should have vir in actual power, but just in case...
            continue
        rbus = rmaster.loc[rmaster['rid'] == rid, 'ResourceBus'].values[0]
        prodtype = record.keys[2]
        if prodtype.upper() == 'EN':
            power_dict = update_power_dict(power_dict, record, rid, rtype, rbus, by_ba)
            found_rids += [rid]
    # Next do power flow on each line:
    flow = mkt_results['p_flow']
    found_tlines = []
    for i in range(flow.number_records):
        fl_rec = flow.next()
        bus1, bus2 = fl_rec.keys[0], fl_rec.keys[1]
        # Only save the current time
        # Note, this assumes keys[2], the 'circuit' is always 1 (only 1 line connecting each)
        if fl_rec.keys[3] == this_time_str:
            power_dict = update_power_dict(power_dict, fl_rec, bus1, bus2, None, False,
                                           transmission=True)
            found_tlines += [(bus1, bus2)]
    # Check and see if any are missing from the GAMS file. If so, add in with 0 power
    rids = rmaster.loc[:,'rid']
    rtypes = rmaster.loc[:,'ResourceType']
    if by_ba:
        with open('ba_names.json', 'r') as f:
            ba_names = json.load(f)
        ba_list = sorted(ba_names['ba'])
        for ba in ba_list:
            for cname in ['power', 'demand']:
                if f'{cname}_{ba}' not in power_dict.keys():
                    power_dict[f'{cname}_{ba}'] = 0
    else:
        for i, r in enumerate(rids):
            if r not in found_rids:
                cname = 'power'
                if rtypes[i] == 'dem':
                    cname = 'demand'
                power_dict[f'{cname}_{rid}'] = 0        
    # Check for missing transmission lines (hardcoding in transmission source for now...)
    tlines = pd.read_excel('market_clearing/system_data/Transmission.xlsx', sheet_name='Clean_BA_Format')
    tlines = tlines['transmission_line'].values
    for tline in tlines:
        bas = tline.split('_')
        # Check for any of the TH_* BAs - if so, join back into a length two list
        bas = handle_TH(bas)
        if f'{bas[0]}_{bas[1]}' in power_dict.keys() or f'{bas[1]}_{bas[0]}' in power_dict.keys():
            continue
        else:
            power_dict.update({f'{bas[0]}_{bas[1]}':0})
    # Sort dictionary (alphabetical order)
    power_keys = list(power_dict.keys())
    power_keys.sort()
    power_tmp = {key: power_dict[key] for key in power_keys}
    # Make sure Time column is first
    power_dict = {'Time': this_time}
    power_dict.update(power_tmp)
    if not newrun:
        header = None
    else:
        header = [h for h in power_dict.keys()]
    power_line = [p for p in power_dict.values()]
    append_csv('system_power.csv', power_line, header=header, savedir=savedir, run_dir=run_dir)

def save_pow_soc(uid, rmaster, newrun=False, save_temp=True, savedir='saved', run_dir='.',
                 save_sys=False):
    ''' This saves the power, soc, and (optionally) temperature to file.
        If save_sys = True it also saves power for the system generators and demand
    '''
    if save_sys:
        save_mkt_pow(uid, rmaster, newrun=newrun, savedir=savedir, run_dir=run_dir)
    this_time = uid_to_time(uid, to_datetime=True)
    disp_dir = join(run_dir, 'physical_dispatch/storage/results')
    ws = gams.GamsWorkspace(working_directory=os.getcwd())
    disp_results = ws.add_database_from_gdx(os.path.join(disp_dir,f'dispatch_{uid}.gdx'))
    # First get power (there will be no record for zero dispatch)
    rids = rmaster.loc[:,'rid']
    rtypes = rmaster.loc[:,'ResourceType']
    power = [rec.value for rec in disp_results["dispatch"]]
    power_rids = [rec.keys[0] for rec in disp_results["dispatch"]]
    power_line = [this_time]
    header = ['Time']
    for i, rid in enumerate(rids):
        if rtypes[i] != 'str':
            continue
        if rid in power_rids:
            rec_idx = power_rids.index(rid)
            power_line += [power[rec_idx]]
        else:
            power_line += [0]
        header += [f'power_{rid}']
    if not newrun:
        header = None
    append_csv('power.csv', power_line, header=header, savedir=savedir, run_dir=run_dir)
    # Now do soc and temp
    soc = disp_results["soc"]
    temp = disp_results["temp"]
    soc_line = [this_time]
    header = ['Time']
    for i in range(soc.number_records):
        record = soc.next()
        rid = record.keys[0]
        soc_line += [record.value]
        header += [f'soc_{rid}']
    if not newrun:
        header = None
    append_csv('soc.csv', soc_line, header=header, savedir=savedir, run_dir=run_dir)
    if save_temp:
        temp_line = [this_time]
        header = ['Time']
        for i in range(temp.number_records):
            record = temp.next()
            rid = record.keys[0]
            temp_line += [record.value]
            header += [f'temp_{rid}']
        if not newrun:
            header = None
        append_csv('temp.csv', temp_line, header=header, savedir=savedir, run_dir=run_dir)

def save_lmp(uid, newrun, savedir='saved', run_dir='.'):
    # Last, save LMP history to file
    this_time = uid_to_time(uid, to_datetime=True)
    str_time = uid_to_time(uid)
    mkt_dir = join(run_dir, 'market_clearing/results')
    ws = gams.GamsWorkspace(working_directory=os.getcwd())
    mkt_results = ws.add_database_from_gdx(os.path.join(mkt_dir,f'results_{uid}.gdx'))
    bus_list = [rec.keys[0] for rec in mkt_results['lmp'] if rec.keys[1] == str_time]
    mult = 12 # Need a multiplier by twelve to convert LMP from $/5min to $/hr
    lw_lmp = [rec.value*mult for rec in mkt_results['lw_lmp'] if rec.keys[0] == str_time]
    if abs(lw_lmp[0]) < 0.001:
        lw_lmp[0] = 0
    lmp_list = [rec.value*mult for rec in mkt_results['lmp'] if rec.keys[1] == str_time]
    for i, lmp in enumerate(lmp_list):
        if abs(lmp) < 0.001:
            lmp_list[i] = 0
    newline = [this_time] + lw_lmp + lmp_list
    if newrun:
        header = ['Time'] + ['lw_lmp'] +  bus_list
        append_csv('lmp.csv', newline, header=header, savedir=savedir)
    else:
        append_csv('lmp.csv', newline, savedir=savedir)  

def get_prev_cleared(prev_uid, intervals, run_dir='.', incl_ba_lmp=False):
    ''' Gets pricing information from the last cleared gdx and returns as a dictionary'''
    market_dir = join(run_dir, 'market_clearing/results')
    if not os.path.isfile(join(market_dir,f'results_{prev_uid}.gdx')):
        return None
    ws = gams.GamsWorkspace(working_directory=os.getcwd())
    mkt_results = ws.add_database_from_gdx(os.path.join(market_dir,f'results_{prev_uid}.gdx'))
    times = [rec.keys[0] for rec in mkt_results['lw_lmp_all']]
    lw_lmps = [rec.value for rec in mkt_results['lw_lmp_all']]
    # Switch to positive convention and multiply to an hour norm
    int_idx = 0
    int_cnt = 0
    mult_list = []
    for i in range(len(lw_lmps)):
        dur = intervals[int_idx][1]
        mult = 60./dur # lmp needs to be scaled to a 60 minute baseline for $/MWh
        mult_list += [mult]
        if abs(lw_lmps[i]) < 0.001:
            lw_lmps[i] = 0
        lw_lmps[i] = mult*lw_lmps[i]
        if intervals[int_idx][0]-1 == int_cnt:
            int_idx += 1
            int_cnt = 0
        else:
            int_cnt += 1
    out = {'prev_uid': prev_uid, 'times': times, 'lw_lmp': lw_lmps}
    # Option to add LMPs for each balancing authority
    if incl_ba_lmp:
        with open('ba_names.json', 'r') as f:
            ba_names = json.load(f)
        ba_list = sorted(ba_names['ba'])
        ba_dict = {}
        for ba in ba_list:
            ba_dict[ba] = []
            ba_times = [rec.keys[1] for rec in mkt_results['lmp'] if ba == rec.keys[0]]
            ba_lmp = [rec.value for rec in mkt_results['lmp'] if ba == rec.keys[0]]
            for i, t in enumerate(times):
                if t in ba_times:
                    tidx = ba_times.index(t)
                    if abs(ba_lmp[tidx]) < 0.001:
                        ba_lmp[tidx] = 0
                    ba_dict[ba] += [ba_lmp[tidx]*mult_list[i]]
                else:
                    ba_dict[ba] = 0
        out.update(ba_dict)
    return out
    
def update_system_offers(resources_df, participant_res, uid, prev_uid, intervals,
                         resource_list=['all'], run_dir='.', drw=1, outages=None):
    """ Takes the system resources and updates their offers as needed """
    default_resource_list = ['gen', 'str', 'ren', 'dem']
    # Make sure the input resouce_list is acceptable
    if resource_list == ['all']:
        resource_list = default_resource_list
    elif type(resource_list) is not list:
        if resource_list in default_resource_list:
            resource_list = [resource_list]
        else:
            raise TypeError("input 'resource_list' must be a list or one of the following strings: 'gen', 'str', 'ren', 'dem'")
    else:
        bad_vals = [i for i in resource_list if i not in default_resource_list]
        if len(bad_vals) > 0:
            raise TypeError("input 'resource_list' must contain only the following elements: gen', 'str', 'ren', 'dem'")  
    # Type labels must match sheet names in 'resource_info.xlsx'
    system_offers = {}
    times = get_time_from_system(uid, run_dir=run_dir)
    tnow, mkt_config = uid_to_time(uid, return_mkt_spec=True, to_datetime=True)
    # Load in forecast and previous energy results
    mkt_dir = join(run_dir, 'market_clearing/results')
    disp_dir = join(run_dir, 'physical_dispatch/storage/results')
    ws = gams.GamsWorkspace(working_directory=os.getcwd())
    prev_en_dict = {}
    prev_soc_dict = {}
    if 'DAM' in mkt_config:
        forecast = get_forecast(uid, intervals, run_dir=run_dir, times=times, drw=drw)
        tlast = tnow - datetime.timedelta(days=1)
        tnow_str = tnow.strftime('%Y%m%d%H%M')
        tlast_str = tlast.strftime('%Y%m%d%H%M')
        dam_uid = mkt_config + tlast_str
        try:
            mkt_results = ws.add_database_from_gdx(os.path.join(mkt_dir,f'results_{dam_uid}.gdx'))
            prev_en_dict = {rec.keys[0]:rec.value for rec in mkt_results['fwd_en'] if 
                            rec.keys[1] == tnow_str}
        except gams.control.workspace.GamsException:
            pass
    else:
        new_i0 = (intervals[0][0]-1,intervals[0][1])
        f_intervals = copy.copy(intervals)
        f_intervals[0] = new_i0
        forecast = get_forecast(uid, f_intervals, run_dir=run_dir, times=times[1:], drw=drw)
        actual = get_actual(uid, intervals, rlist=['all'], run_dir=run_dir, times=[times[0]])
        forecast = join_actual_forecast(actual, forecast)
        if prev_uid is not None:
            mkt_results = ws.add_database_from_gdx(os.path.join(mkt_dir,f'results_{prev_uid}.gdx'))
            prev_en_dict = {rec.keys[0]:rec.value for rec in mkt_results['actual'] if rec.keys[2].upper() == 'EN'}
            disp_results = ws.add_database_from_gdx(os.path.join(disp_dir,f'dispatch_{prev_uid}.gdx'))
            prev_soc_dict = {rec.keys[0]:rec.value for rec in disp_results['soc']}
    for rid in list(resources_df['Master'].rid):
        # Check that type is included (str will often be excluded/saved for participants)
        rtype = resources_df['Master'].query(f"rid=='{rid}'")['ResourceType'].values[0]
        if prev_en_dict == {}:
            prev_en = None
        elif f'{rtype}_{rid}' in prev_en_dict.keys():
            prev_en = prev_en_dict[f'{rtype}_{rid}']
            # print(f'For {rtype}_{rid} found previous energy {prev_en}')
        else:
            prev_en = 0
        if prev_soc_dict == {}:
            prev_soc = None
        elif rtype == 'str':
            prev_soc = prev_soc_dict[rid]
        else:
            prev_soc = None
        if rtype in resource_list:
            attr, offer = format_offer(rid, resources_df, participant_res, uid, prev_en, prev_soc,
                                       forecast, run_dir=run_dir, outages=outages)
            if 'attributes' in system_offers.keys():
                system_offers['attributes'].update({f'{rtype}_{rid}': attr})
            else:
                system_offers['attributes'] = {f'{rtype}_{rid}': attr}
            if 'offer' in system_offers.keys():
                system_offers['offer'].update({f'{rtype}_{rid}': offer})
            else:
                system_offers['offer'] = {f'{rtype}_{rid}': offer}
    # Add a dummy virtual offer
    attr, offer = format_offer(rid, resources_df, participant_res, uid, 0, None, forecast,
                               run_dir=run_dir, dummy_vir=True)
    system_offers['attributes'].update({'vir_p00002_AZPS': attr})
    system_offers['offer'].update({'vir_p00002_AZPS': offer})
    # print("    System Offers:", system_offers['offer']['vir_p00001'])
    return system_offers
            
def format_offer(rid, resources_df, participant_res, uid, prev_en, prev_soc=None, forecast=None,
                 run_dir='.', outages=None,dummy_vir=False):
    """ Makes an offer for a particular resource, returning the attributes and offer dicts """
    type_label = {'gen':'Generators', 'str':'Storage', 'ren':'Renewables', 'dem':'Demand'}
    rtype = resources_df['Master'].query(f"rid=='{rid}'")['ResourceType'].values[0]
    gptype = resources_df['Master'].query(f"rid=='{rid}'")['GridpathType'].values[0]
    rbus = resources_df['Master'].query(f"rid=='{rid}'")['ResourceBus'].values[0]
    try:
        rparticipant = [pid for pid, rlist in participant_res.items() if rid in rlist][0]
    except IndexError:
        rparticipant = 'system' # if not participant owned, we call it part of the system
    # if rtype == 'gen':
    #     newoffer = fl.GenOffer(rid, rparticipant, rbus, rtype, resources_df[type_label[rtype]]) 
    # elif rtype == 'str':
    #     newoffer = fl.StrOffer(rid, rparticipant, rbus, rtype, resources_df[type_label[rtype]])
    if rtype == 'ren':
        rforecast = np.array(forecast[gptype][rbus])
        newoffer = fl.RenOffer(rid, rparticipant, rbus, rtype, resources_df[type_label[rtype]],
                                run_dir=run_dir)
        newoffer.update_ren_offer(uid, rforecast, prev_en)
    elif rtype == 'dem':
        # Read in the selected topology configuration
        with open("topology_conf.json", "r") as f:
            topology = json.load(f)
        mv = topology["dem_block"]["mv"]
        mq_pcnt = topology["dem_block"]["mq"]
        dforecast = np.array(forecast[gptype][rbus])
        newoffer = fl.DemOffer(rid, rparticipant, rbus, rtype, resources_df[type_label[rtype]],
                                run_dir=run_dir)
        newoffer.update_dem_offer(uid, dforecast, mv, mq_pcnt, prev_en)
    elif rtype == 'str':
        newoffer = fl.StrOffer(rid, rparticipant, rbus, rtype, resources_df[type_label[rtype]],
                                run_dir=run_dir) 
        newoffer.update_str_offer(uid, prev_soc, prev_en)
    elif rtype == 'gen':
        newoffer = fl.GenOffer(rid, rparticipant, rbus, rtype, resources_df[type_label[rtype]],
                                run_dir=run_dir) 
        newoffer.update_gen_offer(uid, prev_en, outages=outages)
    else:
        newoffer = fl.OfferType(rid, rparticipant, rbus, rtype, resources_df[type_label[rtype]],
                                run_dir=run_dir) 
        newoffer.update_offer(uid, prev_en)
    attr, offer = newoffer.attributes, newoffer.offer
    if dummy_vir:
        viroffer = fl.OfferType('p00002', 'p00002', 'AZPS', 'vir', dummy_data.vir_offer,
                                run_dir=run_dir) 
        viroffer.update_offer(uid)
        attr, offer = viroffer.attributes, viroffer.offer
    return attr, offer

def outage_datetime_to_str(outages, reverse=False):
    ''' Converts any outage datetimes to str. If reverse, converts str to datetime '''
    for genid in outages.keys():
        if outages[genid]['start_time'] != None:
            if reverse:
                outages[genid]['start_time'] = datetime.datetime.strptime(outages[genid]['start_time'], '%Y%m%d%H%M')
                outages[genid]['end_time'] = datetime.datetime.strptime(outages[genid]['end_time'], '%Y%m%d%H%M')
            else:
                outages[genid]['start_time'] = outages[genid]['start_time'].strftime('%Y%m%d%H%M')
                outages[genid]['end_time'] = outages[genid]['end_time'].strftime('%Y%m%d%H%M')
    return outages

def is_outage(p60days, interval=5):
    ''' Finds the probability of an outage on an interval (in minutes) given a 60 day probability
        Then generates a random number to determine if there is an outage. Returns T/F
    '''
    ratio = 60*24*60/interval
    p_int = 1-np.exp((1/ratio)*np.log(1-p60days))
    prand = np.random.rand()
    outage = False
    if prand < p_int:
        outage = True
    return outage

def check_for_outages(outages, uid, run_dir='.'):
    ''' Cycles through generators, runs a simulation to check for new outages, checks if
        existing outages are over.
    '''
    tnow = uid_to_time(uid, to_datetime=True)
    with open(join(run_dir,'topology_conf.json'),'r') as f:
        topology = json.load(f)
    outage_events = topology["outage_events"]
    # Loop through generators and check for new outages
    i = 0
    for genid in outages.keys():
        # Don't do an outage if it's already going. Also, cap at 2 outages for this simulation
        if outages[genid]['in_outage'] == 1 or outages[genid]['num_outages'] >= 2:
            continue
        for event in outage_events.values():
            is_out = is_outage(event[0])
            i+=1
            if is_out:
                print(f"Starting an outage for generator {genid}, duration is {event[1]} hours")
                outages[genid]['in_outage'] = 1
                outages[genid]['num_outages'] += 1
                outages[genid]['start_time'] = tnow
                tend = tnow + datetime.timedelta(hours=event[1])
                outages[genid]['end_time'] = tend
                break
    # Now check generators to see if it is time to end an existing outage
    for genid in outages.keys():
        if outages[genid]['in_outage'] == 1:
            tend = outages[genid]['end_time']
            if tend <= tnow:
                outages[genid]['in_outage'] = 0
                outages[genid]['start_time'] = None
                outages[genid]['end_time'] = None
    return outages

def add_dispatch_target(d0, rid, record, t_env):
    '''Adds the dispatch target and environment to a gams database object'''
    results_par = add_gams_parameter(d0, 'target', domain=['resource'])
    try:
        results_par.add_record(rid).value = record.value #.marginal, .upper, .lower
    except AttributeError:
        results_par.add_record(rid).value = record
    temp_par = add_gams_parameter(d0, 'temp_env', domain=['resource'])
    temp_par.add_record(rid).value = t_env # Right now we are setting constant enclosure temps

def prep_dispatch(uid, prev_uid, storage_df, run_dir='.', t_env=20, update_attr=False):
    '''Saves appropriate data to gdx files for use in battery_dispatch.gms'''
    dispatch_dir = join(run_dir,'physical_dispatch/storage')
    results_dir = join(run_dir,'market_clearing/results')
        
    ws = gams.GamsWorkspace(working_directory=os.getcwd())
    d0 = ws.add_database() # Database for target/temp
    
    # First sift through the latest results and add the target power and temp from each battery
    mkt_results = ws.add_database_from_gdx(os.path.join(results_dir,f'results_{uid}.gdx'))
    power = mkt_results["actual"]
    t0 = uid_to_time(uid)
    # Sort through all the records and pull out the storage resource target dispatch levels
    records_added = []
    for i in range(power.number_records):
        record = power.next()
        rid = record.keys[0].split('_')[1]
        prodtype = record.keys[2]
        if (rid in storage_df['rid'].values.flatten()) and (record.keys[1] == t0) and \
            (prodtype.upper() == 'EN'):
            if rid in records_added:
                raise ValueError(f'Resource {rid} already has target and temp_env assigned')
            add_dispatch_target(d0, rid, record, t_env)
            records_added += [rid]
    # Gams doesn't put anything for zero entries so add a zero target for any missing resources
    for rid in storage_df['rid'].values.flatten():
        if rid not in records_added:
            add_dispatch_target(d0, rid, 0, t_env)
    time_par = add_gams_parameter(d0, 'duration_t', domain=[])
    time_par.add_record().value = 5
    data_dir = os.path.join(dispatch_dir,'data/')
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    d0.export(os.path.join(data_dir,f'target_{uid}.gdx'))
    
    # Then go through battery attributes and add these to gdx
    # We are modeling this as fixed, so we only do this once
    if (not os.path.exists(os.path.join(data_dir, 'storage_attributes.gdx'))) or \
        (update_attr==True):
        d1 = ws.add_database() # Database for fixed battery attributes
        for rid in storage_df['rid'].values.flatten():
            add_gams_set(d1, rid, set_name='resource', set_dimension=1)
            add_gams_set(d1, rid, set_name='str', set_dimension=1)
            for key, value in storage_attributes.items():
                next_par = add_gams_parameter(d1, key, domain=value)
                next_val = decode_numpy(storage_df.query(f"rid=='{rid}'")[key].values[0])
                next_par.add_record(rid).value = next_val
        d1.export(os.path.join(data_dir, 'storage_attributes.gdx'))
                
    # If this is the first run, save the initial values into a results gdx
    if prev_uid is None:
        dispatch_results_dir = os.path.join(dispatch_dir, 'results/')
        if not os.path.isdir(dispatch_results_dir):
            os.makedirs(dispatch_results_dir)
        d2 = ws.add_database()
        soc_par = add_gams_parameter(d2, 'soc', domain=['resource'])
        temp_par = add_gams_parameter(d2, 'temp', domain=['resource'])
        for rid in storage_df['rid'].values.flatten():
            soc_val = decode_numpy(storage_df.query(f"rid=='{rid}'")['soc_begin'].values[0])
            soc_par.add_record(rid).value = soc_val
            temp_par.add_record(rid).value = t_env # Fix to the environment temperature to start
        d2.export(os.path.join(dispatch_results_dir, 'dispatch_pre_start.gdx'))

def find_time_list(uid, mkt_config, num_fwd):
    '''Returns a list of the forward times at five-minute intervals'''
    this_time = uid_to_time(uid)
    t0 = datetime.datetime.strptime(this_time, '%Y%m%d%H%M')
    phys_shift = 0 # Tracker for whether there is a physical interval or not
    if 'PHYS' in mkt_config.interval_types[0]:
        t0 += datetime.timedelta(minutes=5)
        phys_shift = 1
    t_end = t0
    time_list = []
    fwd_rem = num_fwd
    idx_break = [0] # Tracks when the times interval duration changes
    for interval in mkt_config.interval_durations:
        fwd_cnt, minutes = interval[0], interval[1]
        # On the first pass through, reduce the forward count by 1 if there is a physical interval
        if len(idx_break) == 1:
            fwd_cnt -= phys_shift
        if fwd_cnt >= fwd_rem:
            t_end += datetime.timedelta(minutes=minutes*fwd_rem)
            idx_break += [idx_break[-1]+int(fwd_rem*minutes/5)]
            fwd_rem = 0
            break
        else:
            t_end += datetime.timedelta(minutes=minutes*fwd_cnt)
            idx_break += [idx_break[-1]+int(fwd_cnt*minutes/5)]
            fwd_rem -= fwd_cnt
    num_five_min_intervals = int((t_end-t0).total_seconds()/60/5)
    idx_break = idx_break[1:] # Remove dummy 0 in first position
    for i in range(num_five_min_intervals):
        tnow = t0 + datetime.timedelta(minutes=5*i)
        time_list += [tnow.strftime('%Y%m%d%H%M')]
    return time_list, idx_break

def load_settlement_data(uid, resources_df, run_dir='.'):
    '''
    Creates a ledger object tracking every forward and physical transaction by resource
    Loads settlement data into nested dictionaries (by resource and product type)
    '''
    results_dir = join(run_dir,'market_clearing/results')
    disp_dir = join(run_dir,'physical_dispatch/storage/results')
    this_time = uid_to_time(uid)
    ws = gams.GamsWorkspace(working_directory=os.getcwd())
    # Collect latest settlement data to forward to BalanceSheet
    mkt_results = ws.add_database_from_gdx(os.path.join(results_dir,f'results_{uid}.gdx'))
    if 'DAM' not in uid:
        disp_results = ws.add_database_from_gdx(os.path.join(disp_dir,f'dispatch_{uid}.gdx'))
        phys_power = disp_results['dispatch']
    settlement_in = mkt_results['settlement']
    settlement_out = {}
    for record in settlement_in:
        key1, prodtype = record.keys
        rtype = key1.split('_')[0]
        if rtype.lower() == 'vir':
            rid = key1[4:]
        else:
            rid = key1.split('_')[-1]
        # Only save storage and virtual offers
        if rtype.lower() == 'str':
            prod_dict = {prodtype: -record.value}
            settlement_out[rid] = prod_dict
        if rtype.lower() == 'vir':
            settlement_out[rid] = -record.value
    deltas = mkt_results['delta']
    fwd_en = mkt_results['fwd_en']
    lmps = mkt_results['lmp']
    mcps = mkt_results['mcp']
    ledger = {}
    for record in deltas:
        key1, tstamp, prodtype = record.keys
        k = key1.split('_')
        rtype = k[0]
        rid = k[1:]
        qty = record.value
        # Only save storage and virtual offers
        if rtype.lower() == 'str':
            rid = rid[0]
            if prodtype.upper() == 'EN':
                bus = resources_df['Master'].query(f'rid=="{rid}"')['ResourceBus'].values[0]
                price = lmps.find_record([str(bus), tstamp]).value
                # For physical periods use the dispatched power from battery_dispatch.gms
                # (Note, for settlement we use the delta = dispatched power - sum previous forwards)
                if 'DAM' not in uid and tstamp == this_time:
                    try:
                        dispatch = phys_power.find_record(rid).value
                    except gams.control.workspace.GamsException:
                        dispatch = 0
                    try:
                        prev_fwd = fwd_en.find_record([key1,tstamp]).value
                    except gams.control.workspace.GamsException:
                        prev_fwd = 0
                    qty = dispatch - prev_fwd
            else:
                try:
                    price = mcps.find_record([tstamp, prodtype]).value
                except gams.control.workspace.GamsException:
                    price = 0
            ledger = update_nested_list(ledger, [rid, prodtype, tstamp], [(qty, price)])
        elif rtype.lower() == 'vir':
            pid, bus = rid #.split('_')
            price = lmps.find_record([bus, tstamp]).value
            ledger = update_nested_list(ledger, ['virtual',f'{pid}_{bus}',tstamp], [(qty, price)])
    return ledger, settlement_out
    
def update_settlement_gdx(uid, mkt_config, run_dir='.'):
    ''' 
    Takes a GDX market results file and fills out the settlement for any intervals longer than
    5 minutes (15 min and hourly) since these are not populated by the GAMS script.
    Returns the corresponding 'dense' time list from start time to end time in 5 minute increments
    '''
    results_dir = join(run_dir,'market_clearing/results')
    ws = gams.GamsWorkspace(working_directory=os.getcwd())
    # If the particular market wasn't run on five minute intervals, fill in every five minutes
    num_fwd = [t[0] for t in mkt_config.interval_types if t[1].upper() == 'FWD']
    param_names = ['actual', 'delta', 'fwd_en', 'fwd_nsp', 'fwd_rgd', 'fwd_rgu', 'fwd_spr', 'lmp',
                   'mcp', 'schedule', 'settlement', 'lw_lmp', 'lw_lmp_all'] # 'all_power'
    if len(num_fwd) == 0:
        time_list = gdx_to_json(directory=results_dir,filename=f'results_{uid}',
                                param_names=param_names,make_time_list=True)
        return time_list
    time_list, idx_break = find_time_list(uid, mkt_config, num_fwd[0])
    # Don't need to adjust the settlement gdx for TSRTM or MSRTM
    if 'RTM' in uid:
        return time_list
    # Now with the list of times, remake the database, pausing to output the results to a json file
    mkt_results = ws.add_database_from_gdx(os.path.join(results_dir,f'results_{uid}.gdx'))
    d0 = ws.add_database() # Database to save denser forward positions
    dur = mkt_config.interval_durations
    for name in param_names:
        gams_par = mkt_results[name]
        domains = gams_par.get_domains()
        new_par = add_gams_parameter(d0, name, domains)
        last_rtype = None
        for i in range(gams_par.get_number_records()):
            record = gams_par.next()
            rtype = record.keys[0]
            # Reset indices for each unique resource
            if last_rtype is None or rtype != last_rtype:
                b_cnt = 0 # Count of the time of time resolution breaks in the market intervals
                time_idx = 0 # Count of the time of time resolution breaks in the market intervals
            if 'fwd' in name:
                rkeys = record.keys
                if int(record.keys[1]) < int(time_list[0]) or int(record.keys[1]) > int(time_list[-1]):
                    new_par.add_record(record.keys).value = record.value
                for j, t in enumerate(time_list):
                    # Check if previous times are filled or empty
                    if j < time_idx:
                        continue
                    if int(t) < int(record.keys[1]):
                        pass # Ignore empty backfill values, but still advance index counters below
                    # Now fill forward to the next time...
                    elif int(t) >= int(record.keys[1]) and int(t) < int(record.keys[1])+\
                    dur[b_cnt][1]: # and dur[b_cnt][1] > 5:
                        val = record.value
                        rkeys[1] = t
                        new_par.add_record(rkeys).value = val
                    else:
                        continue
                    time_idx += 1
                    last_rtype = rtype
                    if j+1 == idx_break[b_cnt]:
                        b_cnt += 1
            else:
                new_par.add_record(record.keys).value = record.value
    # Now save the json file (for participants) and updated database (for next market clearing)
    d0.export(os.path.join(results_dir,f'results_{uid}.gdx'))
    gdx_to_json(directory=results_dir,filename=f'results_{uid}',param_names=param_names)
    return time_list
    
def setup_logger(log_level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s/utils - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
