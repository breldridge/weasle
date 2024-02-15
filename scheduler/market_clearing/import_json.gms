
$if not set input $set input pglib-uc/rts_gmlc/2020-06-09.gdx

Sets gen, vre, block, t;
alias(t,tt),(block,b),(gen,g),(vre,v);

$GDXIN %input%
$LOAD gen vre block t
$GDXIN

Parameters
* System
load(t)
* Generator
pmax(gen)
pmin(gen)
ru(gen)
rd(gen)
rsu(gen)
rsd(gen)
minrun(gen)
minoff(gen)
initstatus(gen)
su(gen)
nl(gen)
cost_x(gen,b)
cost_y(gen,b)
vre_min(vre,t)
vre_max(vre,t)
;

$GDXIN %input%
$LOAD load pmax pmin ru rd rsu rsd minrun minoff initstatus su cost_x cost_y vre_min vre_max
$GDXIN