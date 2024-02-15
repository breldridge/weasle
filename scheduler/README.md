# espa-comp
This is the top directory for the ESPA-Comp simulation code.

An instance of the simulation is executed by running scheduler.py.
Market characteristics (the chosen configuration, start time, etc.)
are hardcoded inside the file (near the bottom). Later these may be
moved to an external file for easier editing, or made into command
line arguments.

The directory 'market_clearing' and subdirectories hold the supporting
scripts in python and gams to compute all parts of the market clearing.

The directory 'physical_dispatch' and subdirectories hold scripts to
simulate the physics-based dispatch (using target power from the
market clearing unit committment schedules).

The script load_gen_info.py must be run first (just once/any time
updates are made to the generator input excel files). This will convert
the WECC generator data into a format that is recognized by the scheduler.
