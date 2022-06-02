"""
Function to call the DMBEM heat loss code.

Inputs:
    - Excel Sheet with Building Characteristics

Outputs:
    - Maximum building heat loss coefficient, qHVAC_bc_max
"""
import main
qHVAC_bc_max = main.HLC()
