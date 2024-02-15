# load JSON data to GDX

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import json
import gamstransfer as gt
import pandas as pd

def json_to_gdx(directory = '../pglib-uc/rts_gmlc/', file = '2020-06-09'):
    # Use a breakpoint in the code line below to debug your script.

    location = directory + file
    datafile = location + '.json'
    gdxfile = location + '.gdx'
    fln = datafile

    m = gt.Container()

    print(f"read {fln}")
    with open(fln) as f:
        obj = json.load(f)
    #print(f"data sections: {[k for k in obj.keys()]}")

    # JSON blocks
    demand = obj['demand']
    thermal = obj['thermal_generators']
    renewable = obj['renewable_generators']

    t_list = [f"{t}" for t in range(1, obj['time_periods'] + 1)]

    #Generator data
    gen_list = []
    gen_data = dict()
    gd = gen_data
    gd['pmax'] = []
    gd['pmin'] = []
    gd['ru'] = []
    gd['rd'] = []
    gd['rsu'] = []
    gd['rsd'] = []
    gd['minrun'] = []
    gd['minoff'] = []
    gd['initstatus'] = []
    gd['su'] = []
    cost_x_df = pd.DataFrame(columns=['gen', 'block', 'mw'])
    cost_y_df = pd.DataFrame(columns=['gen', 'block', 'cost'])

    for g, par in thermal.items():
        gen_list.append(par['name'])
        gd['pmax'].append(par['power_output_maximum'])
        gd['pmin'].append(par['power_output_minimum'])
        gd['ru'].append(par['ramp_up_limit'])
        gd['rd'].append(par['ramp_down_limit'])
        gd['rsu'].append(par['ramp_startup_limit'])
        gd['rsd'].append(par['ramp_shutdown_limit'])
        gd['minrun'].append(par['time_up_minimum'])
        gd['minoff'].append(par['time_down_minimum'])
        gd['initstatus'].append(par['unit_on_t0'])
        gd['su'].append(par['startup'][0]['cost'])      # todo: revisit time-dependent start-up costs
        for idx in range(0, len(par['piecewise_production'])):
            x = pd.DataFrame({'gen': g, 'block': idx + 1, 'mw': par['piecewise_production'][idx]['mw']}, index=[0])
            y = pd.DataFrame({'gen': g, 'block': idx + 1, 'cost': par['piecewise_production'][idx]['cost']}, index=[0])
            cost_x_df = pd.concat([cost_x_df, x], ignore_index=True)
            cost_y_df = pd.concat([cost_y_df, y], ignore_index=True)

    # VRE Data
    vre_list = []
    vre_min = pd.DataFrame(columns=['vre', 't', 'vre_min'])
    vre_max = pd.DataFrame(columns=['vre', 't', 'vre_max'])
    for v, par in renewable.items():
        vre_list.append(par['name'])
        par_vmin = par['power_output_minimum']
        par_vmax = par['power_output_maximum']
        for idx in range(0, len(par_vmin)):
            vmin = pd.DataFrame({'vre': v, 't': t_list[idx], 'vre_min': par_vmin[idx]}, index=[0])
            vmax = pd.DataFrame({'vre': v, 't': t_list[idx], 'vre_max': par_vmax[idx]}, index=[0])
            vre_min = pd.concat([vre_min, vmin], ignore_index=True)
            vre_max = pd.concat([vre_max, vmax], ignore_index=True)

    # GAMS Sets
    gen = gt.Set(m, "gen", records=gen_list)
    vre = gt.Set(m, "vre", records=vre_list)
    block = gt.Set(m, "block", records=sorted(cost_x_df['block'].unique()))
    t = gt.Set(m, "t", records=t_list)

    # GAMS Parameters
    m.addParameter('load', domain=t, records=zip(t_list, demand))
    for symname, symdata in gd.items():
        m.addParameter(symname, domain=gen, records=zip(gen_list, symdata))
    m.addParameter('cost_x', domain=[gen, block], records=cost_x_df)
    m.addParameter('cost_y', domain=[gen, block], records=cost_y_df)
    m.addParameter('vre_min', domain=[vre, t], records=vre_min)
    m.addParameter('vre_max', domain=[vre, t], records=vre_max)

    print(f"write {gdxfile}")
    m.write(gdxfile)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    from datasets import datasets
    for folder, files in datasets.items():
        for f in files:
            json_to_gdx(directory=folder, file=f)

