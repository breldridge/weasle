# load XLSX data to GDX

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from gams import *
import gamstransfer as gt
import pandas as pd
import os

domains_1d_params = {
    'hres': 'hyd',
    'hru': 'hyd',
    'hrd': 'hyd'
}

def xlsx_to_gdx(directory = '../Inputs/', file_in = 'wecc_resources', file_out='wecc_resources', index_cols=None):

    datafile = directory + file_in + '.xlsx'
    gdxfile = directory + file_out + '.gdx'
    fln = datafile

    # m = gt.Container()
    ws = GamsWorkspace(working_directory=os.getcwd(), debug=DebugLevel.Off)
    do = ws.add_database()

    print(f"read {fln}")
    if index_cols is not None:
        xls_data = {}
        for sheet, col in index_cols.items():
            xls_data[sheet] = pd.read_excel(fln, sheet_name=sheet, index_col=col)
    else:
        xls_data = pd.read_excel(fln, sheet_name=None)

    # hyd = gt.Set(m, 'hyd', records=par['hyd'].columns.to_list())
    # week = gt.Set(m, 'week', records=par['etot'].index.to_list())

    # add data to GAMS database
    for key, sheet in xls_data.items():
        # Sets
        if key in ['bus', 'gen','hyd', 'bat', 'vre', 't', 'week','costptset', 'pd_block']:
            add_1d_set(do, key, sheet)
        elif key in ['gen2bus','hyd2bus', 'bat2bus', 'vre2bus', 't2week', 'vretype']:
            add_set2bus(do, key, sheet)
        elif key in ['planning_week']:
            add_subset(do, key, sheet, subset_of='week')

        # 1-D Parameters
        elif key in ['hres', 'hru', 'hrd', 'eff', 'cmax', 'dmax', 'socmax', 'socinit', 'sru', 'srd', 'vcost', 'rc', 'drmax',
                     'numcostpts', 'pmin', 'pmax', 'ru', 'rd', 'rsu', 'rsd', 'minrun', 'minoff', 'su', 'sd', 'nl']:
            if key in domains_1d_params.keys():
                add_1d_param(do, key, sheet, domain=do[domains_1d_params[key]])
            else:
                add_1d_param(do, key, sheet)

        # 2-D Parameters
        elif key in ['etot', 'hmax', 'hmin', 'pd', 'vmin', 'vmax', 'cost_x', 'cost_y', 'pd_quant', 'mv']:
            add_2d_param(do, key, sheet)


    print(f"write {gdxfile}")
    do.export(gdxfile)

def add_1d_set(database, key, sheet):
    print(f'loading {key}...', end='')
    gms_set = database.add_set(key, 1)
    for rec in sheet.columns.to_list():
        gms_set.add_record(str(rec))
    print('complete')

def add_set2bus(database, key, sheet):
    print(f'loading {key}...', end='')
    gms_set = database.add_set(key, 2)
    for row in sheet.iterrows():
        gms_set.add_record([str(row[1][0]), str(row[1][1])])
    print('complete')

def add_subset(database, key, sheet, subset_of):
    print(f'loading {key}...', end='')
    gms_set = database.get_symbol(subset_of)
    sub_set = database.add_set_dc(key, [gms_set])
    for rec in sheet.columns.to_list():
        sub_set.add_record(rec).value = 1
    print('complete')

def add_1d_param(database, key, sheet, domain=None):
    print(f'loading {key}...', end='')
    if domain is None:
        gms_par = database.add_parameter(key, 1)
    else:
        gms_par = database.add_parameter_dc(key, [domain])
    for resource in sheet.columns.to_list():
        gms_par.add_record(str(resource)).value = float(sheet[resource].values[0])
    print('complete')

def add_2d_param(database, key, sheet):
    print(f'loading {key}...', end='')
    gms_par = database.add_parameter(key, 2)
    for resource in sheet.columns.to_list():
        for idx in sheet.index.to_list():
            gms_par.add_record([str(resource), str(idx)]).value = float(sheet.loc[idx, resource])
    print('complete')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    index_cols = {
        'bus': None,
        'hyd': None,
        'hyd2bus': None,
        'etot': 'Week',
        'hmax': 'Week',
        'hmin': 'Week',
        'hres': None,
        'hru': None,
        'hrd': None,
        'bat': None,
        'bat2bus': None,
        'eff': None,
        'cmax': None,
        'dmax': None,
        'socmax': None,
        'socinit': None,
        'sru': None,
        'srd': None,
        'pd': 't',
        'pd_block': None,
        'pd_quant': 'block',
        'mv': 'block',
        'rc': None,
        'drmax': None,
        'vmin': 't',
        'vmax': 't',
        'vcost': None,
        'vre': None,
        'vretype': None,
        'vre2bus': None,
        'gen': None,
        'gen2bus': None,
        'costptset': None,
        'numcostpts': None,
        'pmin': None,
        'pmax': None,
        'ru': None,
        'rd': None,
        'rsu': None,
        'rsd': None,
        'minrun': None,
        'minoff': None,
        'su': None,
        'sd': None,
        'nl': None,
        'cost_x': 'costptset',
        'cost_y': 'costptset',
        't': None,
        'week': None,
        't2week': None,
        'planning_week': None,
    }
    xlsx_to_gdx(index_cols=index_cols)

