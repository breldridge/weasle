# load UW's GDX format to my GDX format

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from gams import *
import subprocess, os
import argparse
import gamstransfer as gt
import pandas as pd
import copy
from itertools import product

def gdx_to_gdx(file = 'case5', input_dir = '../datasets/Testcases_gdx/original/',
               output_dir='../datasets/Testcases_gdx/parsed/',
               make_timeseries=False, make_battery=False, make_reserves=False, make_vre=False):
    # Use a breakpoint in the code line below to debug your script.

    gdxin = input_dir + file + '.gdx'
    gdxout = output_dir + file + '_'
    input_args = {}
    input_args['t'] = make_timeseries
    input_args['b'] = make_battery
    input_args['r'] = make_reserves
    input_args['v'] = make_vre
    for i, flag in input_args.items():
        if flag:
            gdxout += i
    if not any(input_args.values()):
        gdxout = gdxout[:-1]
    gdxout += '.gdx'

    if make_vre and not make_timeseries:
        raise Exception("VRE cannot be added without including timeseries demand data (option -t)")

    ws = GamsWorkspace(working_directory=os.getcwd(), debug=DebugLevel.Off)

    print(f"reading {gdxin}")
    di = ws.add_database_from_gdx(gdxin)
    do = ws.add_database()

    # parse sets
    bus = do.add_set("bus", 1)
    for rec in di["bus"]:
        bus.add_record(rec.keys)
    gen = do.add_set("gen", 1)
    for rec in di["gen"]:
        gen.add_record(rec.keys)
    gen2bus = do.add_set("gen2bus",2)
    for rec in di["atBus"]:
        gen2bus.add_record(rec.keys)
    line = do.add_set("line", 3)
    for rec in di["line"]:
        line.add_record(rec.keys)
    circuit = do.add_set("circuit", 1)
    for rec in di["circuit"]:
        circuit.add_record(rec.keys)
    costpts = do.add_set("costptset", 1)
    for rec in di["costptset"]:
        costpts.add_record(rec.keys)

    # Optional set instantiation
    bat = do.add_set('bat',1)
    do.add_set('bat2bus',2)
    vre = do.add_set('vre',1)
    do.add_set('vre2bus',2)

    t = do.add_set("t", 1)
    if make_timeseries:
        for i in range(1,25):
            t.add_record(str(i))
    else:
        for rec in di["t"]:
            t.add_record(rec.keys)

    # BaseMVA
    baseMVA = do.add_parameter('baseMVA',0)
    baseMVA.add_record().value = di['baseMVA'].first_record().value

    # Optional parameter instantiation
    do.add_parameter('respenalty', 0)
    #do.get_parameter('respenalty').add_record().value = 0
    param_dict = {}
    param_dict['resreq'] = [t]
    param_dict['eff'] = [bat]
    param_dict['cmax'] = [bat]
    param_dict['dmax'] = [bat]
    param_dict['socmax'] = [bat]
    param_dict['socinit'] = [bat]
    param_dict['sv'] = [bat]
    param_dict['vmin'] = [vre, t]
    param_dict['vmax'] = [vre, t]
    for name, domain in param_dict.items():
        do.add_parameter_dc(name, domain)

    parse_bus_data(di['businfo'], do, make_timeseries=make_timeseries)
    parse_gen_data(di['geninfo'], do)
    parse_branch_data(di['branchinfo'], do)

    if make_battery:
        create_battery(do)
    if make_reserves:
        create_reserves(do)
    if make_vre:
        create_vre(do)

    # send to gdx?
    print(f"writing {gdxout}")
    do.export(gdxout)

    return print('success!')

def parse_bus_data(businfo, data_out, print_out=False, make_timeseries=False):
    # Bus parameter data
    do = data_out
    bus = do['bus']
    t = do['t']
    baseMVA = do['baseMVA'].first_record()
    adj = baseMVA.value
    bus_list = []
    bus_param_map = {
        ('Pd',None):'pd',
        ('type','given'):'type',
        ('Area','given'):'area'
    }
    bus_data = dict()
    bd = bus_data
    bd['pd'] = {}
    bd['type'] = {}
    bd['area'] = {}
    for rec in businfo:
        b, par, idx = rec.keys
        if b not in bus_list:
            bus_list.append(b)

        if par in ['Pd']:
            name = bus_param_map[(par,None)]
            if b not in bd[name].keys():
                bd[name][b] = {}
            bd[name][b][idx] = rec.value
        elif (par,idx) in bus_param_map.keys():
            bd[bus_param_map[(par,idx)]][b] = rec.value

    # put data into database
    for key in bd.keys():
        if key in ['type','area']:
            par = do.add_parameter_dc(key, [bus])
            for b, val in bd[key].items():
                par.add_record(b).value = val
        elif key in ['pd']:
            if make_timeseries:
                bd['pd'] = calc_timeseries(bd['pd'],file='../datasets/timeseries_load.txt')
            par = do.add_parameter_dc(key, [bus, t])
            for b, time_param in bd[key].items():
                for t_idx, t_val in time_param.items():
                    par.add_record([b, t_idx]).value = t_val / adj

    result = pd.DataFrame(bd, index=bus_list)

    if print_out:
        return print(result)
    else:
        return print(f"parsed bus data for {len(result)} buses")

def parse_gen_data(geninfo, data_out, print_out=False):
    # Generator parameter data
    do = data_out
    gen = do['gen']
    costpts = do['costptset']
    baseMVA = do['baseMVA'].first_record()
    gen_list = []
    gen_param_map = {
        ("Pmax","given"):["pmax"],
        ("Pmin","uwcalc"):["pmin","rsu","rsd"],
        ("RampUp","uwcalc"):["ru"],
        ("RampDown","uwcalc"):["rd"],
        ("MinUptime","given"):["minrun"],
        ("MinDowntime","given"):["minoff"],
        ("startupcost","given"):["su"],
        ("shutdowncost","given"):["sd"],
        ("noloadcost","given"):["nl"],
        ("costpts_x",None):"cost_x",
        ("costpts_y",None):"cost_y",
        ("numcostpts","given"):["numcostpts"]
    }
    gen_data = dict()
    gd = gen_data
    gd['pmax'] = {}
    gd['pmin'] = {}
    gd['ru'] = {}
    gd['rd'] = {}
    gd['rsu'] = {}
    gd['rsd'] = {}
    gd['minrun'] = {}
    gd['minoff'] = {}
    gd['initstatus'] = {}
    gd['su'] = {}
    gd['sd'] = {}
    gd['nl'] = {}
    gd['cost_x'] = {}
    gd['cost_y'] = {}
    gd['numcostpts'] = {}
    for rec in geninfo:
        g, par, idx = rec.keys
        if g not in gen_list:
            gen_list.append(g)

        if par in ['costpts_x','costpts_y']:
            name = gen_param_map[(par,None)]
            if g not in gd[name].keys():
                gd[name][g] = []
            gd[name][g].append(rec.value)
        elif (par, idx) in gen_param_map.keys():
            for name in gen_param_map[(par, idx)]:
                gd[name][g] = rec.value

    # check costpts
    for g in gen_list:
        for key in ['cost_x','cost_y']:
            if len(gd[key][g]) < gd['numcostpts'][g]:
                gd[key][g] = [0] + gd[key][g]
            num_pts = len(gd[key][g])
            num_pts_needed = gd['numcostpts'][g]
            assert num_pts == gd['numcostpts'][g], f'{key} has {num_pts} values but numcostpts is {num_pts_needed}'

    # adjust for BaseMVA
    for key in ['pmax','pmin','ru','rd','rsu','rsd']:
        gd[key] = {k:v/baseMVA.value for k,v in gd[key].items()}
    for g in gd['cost_x'].keys():
        gd['cost_x'][g] = [v/baseMVA.value for v in gd['cost_x'][g]]

    # put data into database
    for key in gd.keys():
        if key in ['pmax','pmin','ru','rd','rsu','rsd','minrun','minoff','initstatus','su','sd','nl','numcostpts','gen2bus']:
            par = do.add_parameter_dc(key, [gen])
            for g, val in gd[key].items():
                par.add_record(g).value = val
        elif key in ['cost_x','cost_y']:
            par = do.add_parameter_dc(key, [gen, costpts])
            for g, pts_list in gd[key].items():
                costpt_set = [c.keys[0] for c in costpts]
                for c in costpt_set[:len(pts_list)]:
                    par.add_record([g, c]).value = pts_list[int(c)-1]

    result = pd.DataFrame(gd, index=gen_list)

    if print_out:
        return print(result)
    else:
        return print(f"parsed gen data for {len(result)} generators")

def parse_branch_data(branchinfo, data_out, print_out=False):
    # Branch parameter data
    do = data_out
    bus_set = do['bus']
    circuit = do['circuit']
    baseMVA = do['baseMVA'].first_record()
    adj = baseMVA.value
    branch_list = []
    branch_param_map = {
        ('r','given'):['r'],
        ('x','given'):['x'],
        ('bc','given'):['bc'],
        ('rateA','given'):['tmax'],
        ('ratio','given'):['ratio']
    }
    branch_data = dict()
    bd = branch_data
    bd['r'] = {}
    bd['x'] = {}
    bd['bc'] = {}
    bd['tmax'] = {}
    bd['ratio'] = {}

    for rec in branchinfo:
        to, fr, cir, par, idx = rec.keys
        line = (to, fr, cir)
        if line not in branch_list:
            branch_list.append(line)
        if (par, idx) in branch_param_map.keys():
            for name in branch_param_map[(par, idx)]:
                bd[name][line] = rec.value

    # adjust for BaseMVA
    for key in ['tmax']:
        bd[key] = {k:v/adj for k,v in bd[key].items()}

    # put data into database
    for key in bd.keys():
        if key in ['r','x','bc','tmax','ratio']:
            par = do.add_parameter_dc(key, [bus_set, bus_set, circuit])
            for b, val in bd[key].items():
                par.add_record(b).value = val

    result = pd.DataFrame(bd, index=branch_list)
    if print_out:
        return print(result)
    else:
        return print(f"parsed branch data for {len(result)} lines")

def calc_timeseries(input_data:dict, file='../datasets/timeseries_load.txt'):
    with open(file, 'r') as f:
        timeseries = [float(val.strip()) for val in f]

    output_data = {}
    timeseries_max = max(timeseries)
    normseries = [val/timeseries_max for val in timeseries]
    new_keys = [str(i) for i in range(1, len(timeseries)+1)]
    for k, param in input_data.items():
        input_max = max(param.values())
        new_series = [input_max*norm for norm in normseries]
        output_data[k] = dict(zip(new_keys, new_series))

    return output_data

def create_battery(gms_database):

    do = gms_database
    type = do['type']
    bat = do.get_set('bat')
    bat2bus = do.get_set('bat2bus')
    baseMVA = do['baseMVA'].first_record()
    adj = baseMVA.value

    # assign the battery to a bus location
    bat.add_record('bat1')
    for rec in type:
        if rec.value == 1:
            b = list(bat.first_record().keys)
            bat_bus = list(rec.keys)
            b.append(bat_bus[0])
            bat2bus.add_record(b)
            break

    # assign storage parameters
    # --- Note: right now there is a single battery with one set of attributes.
    #     If more storage devices are added, will need to decide whether to use
    #     a standard set of parameters (like below), or to assign device-specific
    #     storage attributes.
    bat_dict = {}
    bd = bat_dict
    bd['eff'] = 1
    bd['cmax'] = 10
    bd['dmax'] = 10
    bd['socmax'] = 20
    bd['socinit'] = 0
    bd['sv'] = 10

    # adjust for BaseMVA
    for key in ['sv']:
        bd[key] *= adj

    #create and add parameter in GAMS database
    for key,val in bd.items():
        par = do.get_parameter(key)
        par.add_record('bat1').value = val

    return print(f'added storage resource {b[0]} at bus {b[1]}')

def create_reserves(gms_database):

    do = gms_database
    time = do['t']
    bus = do['bus']
    pd = do['pd']
    pd_keys = [p.keys for p in pd]
    total_demand = [sum(pd[b.keys[0], t.keys[0]].value for b in bus if [b.keys[0], t.keys[0]] in pd_keys) for t in time]
    baseMVA = do['baseMVA'].first_record()
    adj = baseMVA.value

    # default values for reserve parameters
    reserve_dict = {}
    rd = reserve_dict
    rd['respenalty'] = 1000
    rd['resreq'] = 0.10 * max(total_demand)

    # create GAMS parameters for reserve requirement (timeseries) and reserve shortfall penalty (scalar)
    # --- reserve shortfall penalty
    respenalty = do.get_parameter('respenalty')
    respenalty.add_record().value = rd['respenalty'] * adj
    # --- reserve requirements
    resreq = do.get_parameter('resreq')
    for t in time:
        resreq.add_record(t.keys[0]).value = rd['resreq']

    return print(f'added reserve requirements')

def create_vre(gms_database):

    do = gms_database
    type = do['type']
    vre = do.get_set('vre')
    vre2bus = do.get_set('vre2bus')
    baseMVA = do['baseMVA'].first_record()
    adj = baseMVA.value

    # assign the vre to a bus location
    vre.add_record('wind1')
    for rec in type:
        if rec.value == 1:
            v = list(vre.first_record().keys)
            vre_bus = list(rec.keys)
            v.append(vre_bus[0])
            vre2bus.add_record(v)
            break

    vre_dict = {}
    vd = vre_dict
    vd['vmax'] = {'wind1': {'1': 100}}
    vd['vmin'] = {'wind1': {'1': 0}}
    vd['vmax'] = calc_timeseries(vd['vmax'], file='../datasets/timeseries_wind.txt')
    vd['vmin'] = calc_timeseries(vd['vmin'], file='../datasets/timeseries_wind.txt')

    # adjust for BaseMVA
    for name in ['vmin','vmax']:
        for v_idx in vd[name].keys():
            vd_copy = vd[name][v_idx].copy()
            vd[name][v_idx] = {t: val/adj for t, val in vd_copy.items()}

    #create and add parameter in GAMS database
    for var_name, resource in vd.items():
        par = do.get_parameter(var_name)
        for r_idx, series in resource.items():
            for t_idx, val in series.items():
                par.add_record([r_idx, t_idx]).value = val

    return print(f'added wind resource {v[0]} at bus {v[1]}')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--timeseries", action='store_true')
    parser.add_argument("-b", "--battery", action='store_true')
    parser.add_argument("-r", "--reserves", action="store_true")
    parser.add_argument("-v", "--vre", action="store_true")

    args = parser.parse_args()

    gdx_to_gdx(make_timeseries=args.timeseries,
               make_battery=args.battery,
               make_reserves=args.reserves,
               make_vre=args.vre,
               )

