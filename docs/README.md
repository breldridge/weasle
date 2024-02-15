# WEASLE Documents

## Competition Overview
This document describes the overall design of ESPA-Comp at a high level. It describes
the main goals and structure of the competition. Participants may read this guide to 
familiarize themselves with the three market design divisions and simulated market clearing 
procedures before reading the more detailed descriptions provided in the `Participation
Guide` and `Model Formulations` documents.

Storage resource locations and a topology diagram are provided in Section 6 of this document.
Participants may use this information to select the location of their storage resource.

## Participcation Guide
This document describes the logistical details of how a team will participate in 
ESPA-Comp. It includes an in-depth explanation of the algorithm requirements, 
including information on the computing environment, available input data (market 
intervals, forecast and historical data, resource status, and financial settlements), 
and offer format specifications required for the WEASLE simulation platform. This 
document covers the algorithm submission process, including the differences between 
sandbox trial and competition algorithm submissions. It includes a section on how 
scores will be computed and a section with tips and suggestions for troubleshooting 
algorithm submissions. Participants can use this document to better understand how to
use the scripts located in `competitor-tools`.

## Model Formulations
This document provides various model formulations that will are implemented in ESPA-Comp. 
It is not required for potential competitors to read this document, but doing so may be 
helpful to better understand the market clearing process, The market designs, and how 
resource dispatch and degradation will be modeled. This competition will assess the 
performance of different storage offer algorithms in terms of their ability to maximize 
the value of storage resources participating under three market designs that vary in 
complexity. The model formulations include a general market clearing optimization model 
and three market design specifications. A detailed physical dispatch model is used to 
simulate each storage resource’s ability to maintain its scheduled dispatch according to 
its state-of-charge, operating temperature, and other physical attributes.