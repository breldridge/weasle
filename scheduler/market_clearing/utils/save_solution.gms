
*** Save solution
$if not set outfile $set outfile '%run_path%/results/results_%market_uid%.gdx'

Sets
prodtype /EN,RGD,RGU,SPR,NSP/;
Parameter
lmp(i,t)                Energy price at bus i in interval t
lw_lmp(t)               Load-weighted system LMP in interval t (physical only)
lw_lmp_all(t)           Load-weighted system LMP in interval t for all phys fwd and adv intervals
mcp(t,prodtype)         Market clearing price of reserve product in interval t
schedule(n,t,prodtype)  Scheduled dispatch (forward) of resource n at time t for product prodtype
actual(n,t,prodtype)    Actual dispatch (physical) of resource n at time t for product prodtype
all_power(n,t,prodtype) All cleared power quantites (phys fwd adv) of resource n at time t for product prodtype
delta(n,t,prodtype)     Market clearing quantity (delta) of resource n at time t for product prodtype
settlement(n,prodtype)  Market revenue paid (or charged) to resource n for product prodtype
;
lmp(i,t) = -c_balance.m(i,t);
lw_lmp(t)$phys(t) = sum(i,lmp(i,t) * sum(n$dem(n), p_en.l(n,t)$resource_at_bus(n,i))) / sum(i, sum(n$dem(n), p_en.l(n,t)$resource_at_bus(n,i)));
lw_lmp_all(t) = sum(i,lmp(i,t) * sum(n$dem(n), p_en.l(n,t)$resource_at_bus(n,i))) / sum(i, sum(n$dem(n), p_en.l(n,t)$resource_at_bus(n,i)));
mcp(t,'RGU') = c_rgu_supply.m(t);
mcp(t,'RGD') = c_rgd_supply.m(t);
mcp(t,'SPR') = c_spr_supply.m(t);
mcp(t,'NSP') = c_nsp_supply.m(t);
schedule(n,t,'EN')$fwd(t) = p_en.l(n,t);
schedule(n,t,'RGU')$fwd(t) = r_rgu.l(n,t);
schedule(n,t,'RGD')$fwd(t) = r_rgd.l(n,t);
schedule(n,t,'SPR')$fwd(t) = r_spr.l(n,t);
schedule(n,t,'NSP')$fwd(t) = r_nsp.l(n,t);
actual(n,t,'EN')$phys(t) = p_en.l(n,t);
actual(n,t,'RGU')$phys(t) = r_rgu.l(n,t);
actual(n,t,'RGD')$phys(t) = r_rgd.l(n,t);
actual(n,t,'SPR')$phys(t) = r_spr.l(n,t);
actual(n,t,'NSP')$phys(t) = r_nsp.l(n,t);
all_power(n,t,'EN') = p_en.l(n,t);
all_power(n,t,'RGU') = r_rgu.l(n,t);
all_power(n,t,'RGD') = r_rgd.l(n,t);
all_power(n,t,'SPR') = r_spr.l(n,t);
all_power(n,t,'NSP') = r_nsp.l(n,t);
delta(n,t,'EN')$(phys(t) or fwd(t)) = p_en_delta.l(n,t);
delta(n,t,'RGU')$(phys(t) or fwd(t)) = r_rgu_delta.l(n,t);
delta(n,t,'RGD')$(phys(t) or fwd(t)) = r_rgd_delta.l(n,t);  
delta(n,t,'SPR')$(phys(t) or fwd(t)) = r_spr_delta.l(n,t); 
delta(n,t,'NSP')$(phys(t) or fwd(t)) = r_nsp_delta.l(n,t);
settlement(n,'EN') =  sum(t$(phys(t) or fwd(t)), delta(n,t,'EN') * sum(i$resource_at_bus(n,i),lmp(i,t)));  
settlement(n,'RGU') = sum(t$(phys(t) or fwd(t)), delta(n,t,'RGU') * mcp(t,'RGU'));
settlement(n,'RGD') = sum(t$(phys(t) or fwd(t)), delta(n,t,'RGD') * mcp(t,'RGD'));
settlement(n,'SPR') = sum(t$(phys(t) or fwd(t)), delta(n,t,'SPR') * mcp(t,'SPR'));
settlement(n,'NSP') = sum(t$(phys(t) or fwd(t)), delta(n,t,'NSP') * mcp(t,'NSP'));
fwd_en(n,t(td))$fwd(t) = fwd_en(n,t) + delta(n,t,'EN');
fwd_rgu(n,t(td))$fwd(t) = fwd_rgu(n,t) + delta(n,t,'RGU');
fwd_rgd(n,t(td))$fwd(t) = fwd_rgd(n,t) + delta(n,t,'RGD');
fwd_spr(n,t(td))$fwd(t) = fwd_spr(n,t) + delta(n,t,'SPR');
fwd_nsp(n,t(td))$fwd(t) = fwd_nsp(n,t) + delta(n,t,'NSP');

execute_unload '%outfile%' , lmp,lw_lmp,lw_lmp_all,mcp,schedule,actual,delta,settlement,fwd_en,fwd_rgu,fwd_rgd,fwd_spr,fwd_nsp,p_flow.l;

$exit

Parameter summary(*),num_hrs,tot_pd,min_lmp(t),lw_lmp(t);
num_hrs = sum(t$planning_t(t), 1);
tot_pd = sum((i,t)$planning_t(t), pd(i,t));
min_lmp(t)$planning_t(t) = smin(bus, c_balance.m(bus,t));
lw_lmp(t)$planning_t(t) = sum(bus, v_pd.l(bus,t) * c_balance.m(bus,t)) / sum(bus, v_pd.l(bus,t));
*** absolute calculations
summary('tx_cong_$') = sum((i,j,c,t)$(line(i,j,c) and planning_t(t)), v_pf.m(i,j,c,t) * v_pf.l(i,j,c,t)) + eps;
summary('tx_vio_mw') =  sum((i,j,c,t)$(line(i,j,c) and planning_t(t)), v_pfslackneg.l(i,j,c,t) + v_pfslackpos.l(i,j,c,t)) + eps;
summary('imbalance_mw') = sum((i,t)$planning_t(t), v_imbalancepos.l(i,t) + v_imbalanceneg.l(i,t)) + eps;
summary('en_curt_mw') = sum((i,t)$planning_t(t), (pd(i,t) - v_pd.l(i,t))$(pd(i,t) > v_pd.l(i,t))) + eps;
summary('en_addl_mw') = sum((i,t)$planning_t(t), (v_pd.l(i,t) - pd(i,t))$(pd(i,t) < v_pd.l(i,t))) + eps;
$if %renewables%==1 summary('wind_curt_mw') = sum((vre,t)$(vretype(vre,'wind') and planning_t(t)), vmax(vre,t) - v_vre.l(vre,t)) + eps;
$if %renewables%==1 summary('solar_curt_mw') = sum((vre,t)$(vretype(vre,'solar') and planning_t(t)), vmax(vre,t) - v_vre.l(vre,t)) + eps;
*** relative calculations
summary('min_lmp_neg%') = sum(t$(planning_t(t) and (min_lmp(t) < 0)), 1) / num_hrs + eps;
summary('min_lmp_npos%') = sum(t$(planning_t(t) and (min_lmp(t) <= 0)), 1) / num_hrs + eps;
summary('lw_lmp_neg%') = sum(t$(planning_t(t) and (lw_lmp(t) < 0)), 1) / num_hrs + eps;
summary('lw_lmp_npos%') = sum(t$(planning_t(t) and (lw_lmp(t) <= 0)), 1) / num_hrs + eps;
summary('tx_cong_mkts%') = (summary('tx_cong_$') / v_mktsurplus.l)$(abs(v_mktsurplus.l) > 1) + eps;
summary('tx_cong_cost%') = (summary('tx_cong_$') / v_gencost.l)$(abs(v_gencost.l) > 1) + eps;
summary('tx_vio_%') = summary('tx_vio_mw') / tot_pd;
summary('imbalance_%') = summary('imbalance_mw') / tot_pd;
summary('en_curt_%') = summary('en_curt_mw') / tot_pd;
summary('en_addl_%') = summary('en_addl_mw') / tot_pd;
$if %renewables%==1 summary('wind_curt_%') = summary('wind_curt_mw') / sum((vre,t)$(vretype(vre,'wind') and planning_t(t)), vmax(vre,t));
$if %renewables%==1 summary('solar_curt_%') = summary('solar_curt_mw') / sum((vre,t)$(vretype(vre,'solar') and planning_t(t)), vmax(vre,t));
$onEcho > writeSummary.txt
epsOut=0 par=summary rng=summary! cdim=0
$offEcho

Sets headings /pd,pg,vre,bat,ch,dc,soc,ph,hyd,phww,hww,lmp,res/;
Parameter dispatch(t,headings);
dispatch(t,'pd')$planning_t(t) = sum(bus, v_pd.l(bus,t));
$if %generators%==1 dispatch(t,'pg')$planning_t(t) = sum(gen, v_pg.l(gen,t)) + eps;
$if %renewables%==1 dispatch(t,'vre')$planning_t(t) = sum(vre, v_vre.l(vre,t)) + eps;
$if %storage%==1 dispatch(t,'bat')$planning_t(t) = sum(bat, v_dc.l(bat,t) - v_ch.l(bat,t)) + eps;
$if %hydro_std%==1 dispatch(t,'hyd')$planning_t(t) = sum(hyd, v_ph.l(hyd,t)) + eps;
$if %hydro_ww%==1 dispatch(t,'hww')$planning_t(t) = sum(hww, v_phww.l(hww,t)) + eps;
dispatch(t,'lmp')$planning_t(t) = sum(bus, c_balance.m(bus,t) * v_pd.l(bus,t)) / dispatch(t,'pd');
$onEcho > writeDispatch.txt
epsOut=0 par=dispatch rng=dispatch! 
$offEcho

Parameter objective(*);
objective('mktsurplus') = v_mktsurplus.l + eps;
objective('imbalpenalty') = v_imbalpenalty.l + eps;
objective('flowpenalty') = v_flowpenalty.l + eps;
objective('respenalty') = v_respenalty.l + eps;
$if %generators%==1 objective('gencost') = v_gencost.l + eps;
$if %renewables%==1 objective('vrecost') = v_vrecost.l + eps;
$if %flex_load%==1 objective('bidvalue') = v_bidvalue.l + eps;
$if %storage%==1 objective('socvalue') = v_socvalue.l + eps;
$if %hydro_std%==1 objective('hydcost') = v_hydcost.l + eps;
$if %hydro_ww%==1 objective('hwwcost') = v_hwwcost.l + eps;
$onEcho > writeObj.txt
epsOut=0 par=objective rng=objective! cdim=0
$offEcho

Parameter prices(t,*);
prices(t,bus)$planning_t(t) = c_balance.m(bus,t) + eps;
prices(t,'reserve')$planning_t(t) = c_resreq.m(t) + eps;
$onEcho > writePrices.txt
hText="hour,LMP,,,,,MCP" rng=prices!
epsOut=0 par=prices rng=prices!A2
$offEcho

Parameter generators(t,*,*);
$ifthen %generators%==1
$set outfile %outfile%G
generators(t,'pg',gen)$planning_t(t) = v_pg.l(gen,t) + eps;
generators(t,'res',gen)$planning_t(t) = v_genres.l(gen,t) + eps;
$endif
$onEcho > writeGens.txt
epsOut=0 par=generators rng=generators! cdim=2
$offEcho

Parameter flex_load(t,*,bus);
$ifthen %flex_load%==1
$set outfile %outfile%L
flex_load(t,'pd',bus)$planning_t(t) = v_pd.l(bus,t) + eps;
flex_load(t,'res',bus)$planning_t(t) = v_demres.l(bus,t) + eps;
flex_load(t,'diff',bus)$planning_t(t) = v_pd.l(bus,t) - pd(bus,t) + eps;
$endif
$onEcho > writeFlexLoad.txt
epsOut=0 par=flex_load rng=flex_load! cdim=2
$offEcho

Parameter storage(t,*,*);
$ifthen %storage%==1
$set outfile %outfile%S
storage(t,'ch',bat)$planning_t(t) = v_ch.l(bat,t) + eps;
storage(t,'dc',bat)$planning_t(t) = v_dc.l(bat,t) + eps;
storage(t,'soc',bat)$planning_t(t) = v_soc.l(bat,t) + eps;
storage(t,'res',bat)$planning_t(t) = v_batres.l(bat,t) + eps;
$endif
$onEcho > writeStorage.txt
epsOut=0 par=storage rng=storage! cdim=2
$offEcho

Parameter renewables(t,*,*);
$ifthen %renewables%==1
$set outfile %outfile%R
renewables(t,'vre',vre)$planning_t(t) = v_vre.l(vre,t) + eps;
renewables(t,'res',vre)$planning_t(t) = v_vreres.l(vre,t) + eps;
$endif
$onEcho > writeRenewables.txt
epsOut=0 par=renewables rng=renewables! cdim=2
$offEcho

Parameter hydro_std(t,*,*);
$ifthen %hydro_std%==1
$set outfile %outfile%H
hydro_std(t,'ph',hyd)$planning_t(t) = v_ph.l(hyd,t) + eps;
hydro_std(t,'res',hyd)$planning_t(t) = v_hydres.l(hyd,t) + eps;
$endIf
$onEcho > writeHydroStd.txt
epsOut=0 par=hydro_std rng=hydro_std! cdim=2
$offEcho

Parameter hydro_ww(t,*,*);
$ifthen %hydro_ww%==1
$set outfile %outfile%W
hydro_ww(t,'ph',hww)$planning_t(t) = v_phww.l(hww,t) + eps;
hydro_ww(t,'res',hww)$planning_t(t) = v_hwwres.l(hww,t) + eps;
$endIf
$onEcho > writeHydroWW.txt
epsOut=0 par=hydro_ww rng=hydro_ww! cdim=2
$offEcho


execute_unload '%outfile%.gdx', objective,summary,prices,dispatch,generators,flex_load,storage,renewables,hydro_std,hydro_ww;
execute 'gdxxrw.exe %outfile%.gdx @writeObj.txt';
execute 'gdxxrw.exe %outfile%.gdx @writeSummary.txt';
execute 'gdxxrw.exe %outfile%.gdx @writePrices.txt';
execute 'gdxxrw.exe %outfile%.gdx @writeDispatch.txt';
$if %generators%==1 execute 'gdxxrw.exe %outfile%.gdx @writeGens.txt';
$if %flex_load%==1 execute 'gdxxrw.exe %outfile%.gdx @writeFlexLoad.txt';
$if %storage%==1 execute 'gdxxrw.exe %outfile%.gdx @writeStorage.txt';
$if %renewables%==1 execute 'gdxxrw.exe %outfile%.gdx @writeRenewables.txt';
$if %hydro_std%==1 execute 'gdxxrw.exe %outfile%.gdx @writeHydroStd.txt';
$if %hydro_ww%==1 execute 'gdxxrw.exe %outfile%.gdx @writeHydroWW.txt';