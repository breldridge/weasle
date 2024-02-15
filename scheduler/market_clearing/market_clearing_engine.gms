*** ESPA-Comp: Market Clearing Engine

$setnames "%gams.i%" filepath filename fileextension
* $set market_uid TSDAM201801280000
* $set previous_uid pre_start
* $set market_uid TSRTM201801280045
* $set previous_uid TSRAM201801280040
$if not set market_uid $abort "Must provide a market UID to initialize the market clearing engine."
$if not set run_path $set run_path '.'
$if not set relax $set relax 0
$if not set savesol $set savesol 1
$if not set network $set network case5

option optcr=0.01;

$include "utils/load_market_data.gms"
$include "utils/load_resource_data.gms"

Parameters
* System
penalty_en              /3000/
value_coeff_rgu         /250/
rgu_ex_max              /100/
value_coeff_rgd         /250/
rgd_ex_max              /100/
value_coeff_spr         /100/
spr_ex_max              /100/
value_coeff_nsp         /500/
nsp_ex_max              /100/
* value_coeff_rgu         /0/
* rgu_ex_max              /0/
* value_coeff_rgd         /0/
* rgd_ex_max              /0/
* value_coeff_spr         /0/
* spr_ex_max              /0/
* value_coeff_nsp         /0/
* nsp_ex_max              /0/
value_coeff_rgu_block(b)
value_coeff_rgd_block(b)
value_coeff_spr_block(b)
value_coeff_nsp_block(b)
pcnt_reg_block1         /0.4/
pcnt_reg_block2         /0.1/
penalty_pflow           /1000/
coeff_rgu               /0.03/
coeff_rgd               /0.03/
coeff_spr               /1.0/
coeff_nsp               /2.0/
duration_rgu            /0/
duration_rgd            /0/
duration_spr            /30/
duration_nsp            /60/
response_spr            /10/
response_nsp            /30/
;

* Set block values to 40% and 10% of value coefficient
value_coeff_rgu_block('1') = value_coeff_rgu*0.4;
value_coeff_rgu_block('2') = value_coeff_rgu*0.1;
value_coeff_rgd_block('1') = value_coeff_rgd*0.4;
value_coeff_rgd_block('2') = value_coeff_rgd*0.1;
value_coeff_spr_block('1') = value_coeff_spr*0.4;
value_coeff_spr_block('2') = value_coeff_spr*0.1;
value_coeff_nsp_block('1') = value_coeff_nsp*0.4;
value_coeff_nsp_block('2') = value_coeff_nsp*0.1;

Variables
* Objective function
z_mktsurplus            Total market surplus objective
z_surplus(n)            Value minus cost accrued to resource n
z_en_excess(i,t)        Penalty for energy excess at bus i in interval t
z_en_shortage(i,t)      Penalty for energy shortage at bus i in interval t
z_rgu_excess(t)         Value of regulation up reserve excess in interval t
z_rgu_shortage(t)       Penalty due to regulation up reserve shortage in interval t
z_rgd_excess(t)         Value of regulation down reserve excess in interval t
z_rgd_shortage(t)       Penalty due to regulation down reserve shortage in interval t
z_spr_excess(t)         Value of spinning reserve excess in interval t
z_spr_shortage(t)       Penalty due to spinning reserve shortage in interval t
z_nsp_excess(t)         Value of non-spinning reserve excess in interval t
z_nsp_shortage(t)       Penalty due to non-spinning reserve shortage in interval t
z_pflow_vio(i,j,c,t)    Penalty due to line overload on line i j c in interval t
* System variables
v_pd(i,t)               Power demand at bus i in interval t
v_theta(i,t)            Bus voltage angle in interval t
p_en_excess(i,t)        Excess energy supply at bus i in interval t
p_en_shortage(i,t)      Excess energy demand at bus i in interval t
p_flow(i,j,c,t)         Power flow on line i j c in interval t
p_flow_vio(i,j,c,t)     Overload on line i j c in interval t
* r_rgu_excess(t)         Regulation up reserve excess in interval t
r_rgu_ex_block(b,t)     Regulation up excess block in interval t
r_rgu_shortage(t)       Regulation up reserve shoratge in interval t
* r_rgd_excess(t)         Regulation down reserve excess in interval t
r_rgd_ex_block(b,t)     Regulation down excess block in interval t
r_rgd_shortage(t)       Regulation down reserve shortage in interval t
* r_spr_excess(t)         Spinning reserve excess in interval t
r_spr_ex_block(b,t)     Spinning reserve up excess block in interval t
r_spr_shortage(t)       Spinning reserve shortage in interval t
* r_nsp_excess(t)         Non-spinning reserve excess in interval t
r_nsp_ex_block(b,t)     Non-spinning reserve up excess block in interval t
r_nsp_shortage(t)       Non-spinning reserve shortage in interval t
r_rgu_req(t)            Regulation up reserve requirement in interval t
r_rgd_req(t)            Regulation down reserve requirement in interval t
r_spr_req(t)            Spinning reserve requirement in interval t
r_nsp_req(t)            Non-spinning reserve requirement in interval t
* Generic Resource Variables
z_en(n)                 Total cost of energy dispatch provided by resource n
z_rgu(n)                Cost to provide regulation up reserve from resource n 
z_rgd(n)                Cost to provide regulation down reserve from resource n
z_spr(n)                Cost to provide spinning reserve from resource n 
z_nsp(n)                Cost to provide non-spinning reserve from resource n
p_en(n,t)               Net energy injection from resource n in interval t
p_en_block(n,b,t)       Auxiliary variable for energy dispatch block b
r_rgu(n,t)              Regulation up reserve provided by resource n in interval t
r_rgd(n,t)              Regulation down reserve provided by resource n in interval t
r_spr(n,t)              Spinning reserve provided by resource n in interval t
r_nsp(n,t)              Non-spinning reserve provided by resource n in interval t
* Storage Variables
z_soc(n)                Cost of stored energy
p_ch(n,t)               Charging level of the storage resource in interval t
p_dc(n,t)               Discharging level of the storage resource in period t
p_ch_block(n,b,t)       Auxiliary dispatch variable for charging quantity block b
p_dc_block(n,b,t)       Auxiliary dispatch variable for discharging quantity block b
e_soc(n,t)              State-of-charge at the end of period t
e_soc_block(n,b,t)      Auxiliary state-of-charge variable for stored energy quantity block b
* Generator Variables
z_su(n)                 Cost of starting up generator n
z_op(n)                 Cost for generator n to remain online
z_sd(n)                 Cost to shut down generator n
* Virtual Variables
p_dec_block(n,b,t)      Auxiliary variable for virtual energy decrement in block b
p_inc_block(n,b,t)      Auxiliary variable for virtual energy increment in block b
* Settlement Variables
p_en_delta(n,t)         Change in energy schedule compared to forward position
r_rgd_delta(n,t)        Change in regulation up reserve schedule compared to forward position
r_rgu_delta(n,t)        Change in regulation down reserve schedule compared to forward position
r_spr_delta(n,t)        Change in spinning reserve schedule compared to forward position
r_nsp_delta(n,t)        Change in non-spinning reserve schedule compared to forward position
;
Binary Variables
u_status(n,t)           Unit commitment status in interval t
u_startup(n,t)          Unit start-up indicator for interval t
u_shutdown(n,t)         Unit shut-down indicator for interval t
;

Equations
c_mktsurplus            Sum of market suplus
c_balance(i,t)          Power balance at bus i in interval t
c_en_excess(i,t)        Real power excess penalty at bus i in interval t
c_en_shortage(i,t)      Real power shortage penalty at bus i in interval t
c_pflow(i,j,c,t)        Power flow on line i j c in interval t
c_pflow_ub(i,j,c,t)     Upper bound for power flow on line i j c in interval t
c_pflow_lb(i,j,c,t)     Lower bound for power flow on line i j c in interval t
c_pflow_vio(i,j,c,t)    Power flow overload penalty on line i j c in interval t
c_rgu_supply(t)         Regulation up reserve supply balance in interval t
c_rgd_supply(t)         Regulation down reserve supply balance in interval t
c_spr_supply(t)         Spinning reserve supply balance in interval t
c_nsp_supply(t)         Non-spinning reserve supply balance in interval t
c_rgu_excess(t)         Value of regulation up reserve excess in interval t            
c_rgu_shortage(t)       Penalty due to regulation up reserve shortage in interval t    
c_rgd_excess(t)         Value of regulation down reserve excess in interval t          
c_rgd_shortage(t)       Penalty due to regulation down reserve shortage in interval t  
c_spr_excess(t)         Value of spinning reserve excess in interval t                 
c_spr_shortage(t)       Penalty due to spinning reserve shortage in interval t         
c_nsp_excess(t)         Value of non-spinning reserve excess in interval t             
c_nsp_shortage(t)       Penalty due to non-spinning reserve shortage in interval t
c_rgu_req(t)            Regulation up reserve requirement in interval t
c_rgd_req(t)            Regulation down reserve requirement in interval t
c_spr_req(n,t)          Spinning reserve requirement in interval t
c_nsp_req(n,t)          Non-spinning reserve requirement in interval t

* rgu_capacity(n,t)       Regulation up capacity of resource n in interval t
* rgd_capacity(n,t)       Regulation down capacity of resource n in interval t
* spr_capacity(n,t)       Spinning reserve capacity of resource n in interval t
* nsp_capacity(n,t)       Non-spinning reserve capacity of resource n in interval t
* rampspr_capacity(n,t)   Spinning reserve ramping capacity up of resource n in interval t
* rampnsp_capacity(n,t)   Non-spinning reserve ramping capacity up of resource n in interval t

c_en_delta(n,t)         Change in energy schedule compared to forward position
c_rgd_delta(n,t)        Change in regulation up reserve schedule compared to forward position
c_rgu_delta(n,t)        Change in regulation down reserve schedule compared to forward position
c_spr_delta(n,t)        Change in spinning reserve schedule compared to forward position
c_nsp_delta(n,t)        Change in non-spinning reserve schedule compared to forward position
;

* Variable bounds
$ifthen %flex_load%==0
v_pd.fx(bus,t) = pd(bus,t);
$endif
* Network variables
v_theta.lo(bus,t) = -inf;
v_theta.up(bus,t) =  inf;
v_theta.fx(bus,t)$(type(bus)=1) = 0;
p_en_excess.lo(i,t) = 0;   
p_en_shortage.lo(i,t) = 0;
p_flow_vio.lo(i,j,c,t) = 0;
* Objective variables
z_en_excess.up(i,t) = 0;     
z_en_shortage.up(i,t) = 0;      
z_rgu_excess.lo(t) = 0;
z_rgd_excess.lo(t) = 0;
z_spr_excess.lo(t) = 0;
z_nsp_excess.lo(t) = 0;
z_rgu_shortage.up(t) = 0;
z_rgd_shortage.up(t) = 0;
z_spr_shortage.up(t) = 0;
z_nsp_shortage.up(t) = 0;
z_pflow_vio.up(i,j,c,t) = 0;
* Reserve variables
r_rgu.lo(n,t) = 0;          
r_rgd.lo(n,t) = 0;          
r_spr.lo(n,t) = 0;          
r_nsp.lo(n,t) = 0;
* r_rgu_excess.up(t) = value_coeff_rgu/(value_coeff_rgu2 + (value_coeff_rgu2=0));
* r_rgd_excess.up(t) = value_coeff_rgd/(value_coeff_rgd2 + (value_coeff_rgd2=0));
* r_spr_excess.up(t) = value_coeff_spr/(value_coeff_spr2 + (value_coeff_spr2=0));
* r_nsp_excess.up(t) = value_coeff_nsp/(value_coeff_nsp2 + (value_coeff_nsp2=0));
r_rgu_ex_block.up(b,t) = rgu_ex_max/2.0;
r_rgd_ex_block.up(b,t) = rgd_ex_max/2.0;
r_spr_ex_block.up(b,t) = spr_ex_max/2.0;
r_nsp_ex_block.up(b,t) = nsp_ex_max/2.0;
r_rgu_ex_block.lo(b,t) = 0;
r_rgd_ex_block.lo(b,t) = 0;
r_spr_ex_block.lo(b,t) = 0;
r_nsp_ex_block.lo(b,t) = 0; 
r_rgu_shortage.lo(t) = 0;   
r_rgd_shortage.lo(t) = 0;   
r_spr_shortage.lo(t) = 0;    
r_nsp_shortage.lo(t) = 0;   
r_rgu_req.lo(t) = 0;
r_rgd_req.lo(t) = 0;
r_spr_req.lo(t) = 0;
r_nsp_req.lo(t) = 0;


* Constraints
*** Objective function
c_mktsurplus..  z_mktsurplus =e=
                    sum(n, z_surplus(n))
                    + sum((i,j,c,t)$monitored(i,j,c), z_pflow_vio(i,j,c,t))
                    + sum((i,t), z_en_excess(i,t) + z_en_shortage(i,t))
                    + sum(t, z_rgu_excess(t) + z_rgu_shortage(t)
                           + z_rgd_excess(t) + z_rgd_shortage(t)
                           + z_spr_excess(t) + z_spr_shortage(t)
                           + z_nsp_excess(t) + z_nsp_shortage(t))
;
* Power balance
c_balance(i,t)..    sum(n$resource_at_bus(n,i), p_en(n,t))
                    - p_en_excess(i,t)
                    + p_en_shortage(i,t)
                    - sum((j,c)$line(i,j,c), p_flow(i,j,c,t))
                    + sum((j,c)$line(j,i,c), p_flow(j,i,c,t))
                    =e= 0
;
c_en_excess(i,t)..      z_en_excess(i,t) =e= -duration_h(t) * penalty_en * p_en_excess(i,t);
c_en_shortage(i,t)..    z_en_shortage(i,t) =e= -duration_h(t) * penalty_en * p_en_shortage(i,t);

* Power flow
c_pflow(i,j,c,t)$line(i,j,c)..      p_flow(i,j,c,t) =e= linestatus(i,j,c) * Bsus(i,j,c)*(v_theta(i,t) - v_theta(j,t));
c_pflow_ub(i,j,c,t)$line(i,j,c)..   p_flow(i,j,c,t) =l= tmax(i,j,c) + p_flow_vio(i,j,c,t);
c_pflow_lb(i,j,c,t)$line(i,j,c)..   -p_flow(i,j,c,t) =l= tmax(i,j,c) + p_flow_vio(i,j,c,t);
c_pflow_vio(i,j,c,t)$line(i,j,c)..  z_pflow_vio(i,j,c,t) =e= -duration_h(t) * penalty_pflow * p_flow_vio(i,j,c,t);

* Reserves
* c_rgu_supply(t)..   r_rgu_excess(t) + r_rgu_req(t) =l= r_rgu_shortage(t) + sum(n, r_rgu(n,t));
c_rgu_supply(t)..   sum(b, r_rgu_ex_block(b,t))  + r_rgu_req(t) =l= r_rgu_shortage(t) + sum(n, r_rgu(n,t));
c_rgd_supply(t)..   sum(b, r_rgd_ex_block(b,t)) + r_rgd_req(t) =l= r_rgd_shortage(t) + sum(n, r_rgd(n,t));
c_spr_supply(t)..   sum(b, r_spr_ex_block(b,t)) + r_rgu_req(t) + r_spr_req(t)
                        =l= r_spr_shortage(t) + sum(n, r_rgu(n,t) + r_spr(n,t));
c_nsp_supply(t)..   sum(b, r_nsp_ex_block(b,t)) + r_rgu_req(t) + r_spr_req(t) + r_nsp_req(t)
                        =l= r_nsp_shortage(t) + sum(n, r_rgu(n,t) + r_spr(n,t) + r_nsp(n,t));
c_rgu_excess(t)..   z_rgu_excess(t) =l= duration_h(t) * sum(b, r_rgu_ex_block(b,t)*value_coeff_rgu_block(b));
c_rgd_excess(t)..   z_rgd_excess(t) =l= duration_h(t) * sum(b, r_rgd_ex_block(b,t)*value_coeff_rgd_block(b));
c_spr_excess(t)..   z_spr_excess(t) =l= duration_h(t) * sum(b, r_spr_ex_block(b,t)*value_coeff_spr_block(b));
c_nsp_excess(t)..   z_nsp_excess(t) =l= duration_h(t) * sum(b, r_nsp_ex_block(b,t)*value_coeff_nsp_block(b));

c_rgu_shortage(t).. z_rgu_shortage(t) =e= -duration_h(t) * value_coeff_rgu * r_rgu_shortage(t);
c_rgd_shortage(t).. z_rgd_shortage(t) =e= -duration_h(t) * value_coeff_rgd * r_rgd_shortage(t);
c_spr_shortage(t).. z_spr_shortage(t) =e= -duration_h(t) * value_coeff_spr * r_spr_shortage(t);
c_nsp_shortage(t).. z_nsp_shortage(t) =e= -duration_h(t) * value_coeff_nsp * r_nsp_shortage(t);
c_rgu_req(t)..      r_rgu_req(t) =g= -coeff_rgu * sum(dem, p_en(dem,t));
c_rgd_req(t)..      r_rgd_req(t) =g= -coeff_rgd * sum(dem, p_en(dem,t));
c_spr_req(n,t)$(gen(n) or ren(n) or str(n))..     r_spr_req(t) =g= coeff_spr * p_en(n,t);
c_nsp_req(n,t)$(gen(n) or ren(n) or str(n))..     r_nsp_req(t) =g= coeff_nsp * p_en(n,t);

* Include resource models
$include "resource_models/generators.gms"
$include "resource_models/renewables.gms"
$include "resource_models/storage.gms"
$include "resource_models/flex_load.gms"
$include "resource_models/virtual.gms"

* Settlement
c_en_delta(n,t)$(phys(t) or fwd(t))..   p_en(n,t) =e= p_en_delta(n,t) + fwd_en(n,t);
c_rgd_delta(n,t)$(phys(t) or fwd(t))..  r_rgd(n,t) =e= r_rgd_delta(n,t) + fwd_rgd(n,t);
c_rgu_delta(n,t)$(phys(t) or fwd(t))..  r_rgu(n,t) =e= r_rgu_delta(n,t) + fwd_rgu(n,t);
c_spr_delta(n,t)$(phys(t) or fwd(t))..  r_spr(n,t) =e= r_spr_delta(n,t) + fwd_spr(n,t);
c_nsp_delta(n,t)$(phys(t) or fwd(t))..  r_nsp(n,t) =e= r_nsp_delta(n,t) + fwd_nsp(n,t);

* Market model
model market /all/;

* option rmiqcp=cplex, miqcp=cplex;
$ifthen %relax%==1
solve market max z_mktsurplus using rmip;
$else
solve market max z_mktsurplus using mip;
$endif

$ifthen %savesol%==0
execute_unload '%run_path%/results/results_%market_uid%.gdx', z_mktsurplus,p_en,r_rgu,r_rgd,r_spr,r_nsp,c_balance,c_rgu_supply,c_rgd_supply,c_spr_supply,c_nsp_supply;
$elseIf %savesol%==1
$include "utils/save_solution.gms"
$endIf