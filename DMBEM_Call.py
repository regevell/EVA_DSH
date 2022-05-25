"""
Function to call the DMBEM heat loss code.

Inputs:
    - Excel Sheet with Building Characteristics

Outputs:
    - Maximum building heat loss coefficient, qHVAC_bc_max
"""

qHVAC_bc_max = main.HLC()
