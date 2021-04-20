import pandas as pd
import numpy as np
from datetime import datetime
import epyestim
import epyestim.covid19 as covid19
import sys, os
import pdb
newdf=pd.read_csv('/project/biocomplexity/aniadiga/gits/google_mobility_v2/regression/csv/daily_case_data.csv',dtype={'FIPS':str})
newdf=newdf.set_index('FIPS')
pdb.set_trace()
newdf.columns=pd.to_datetime(newdf.columns)

cn=sys.argv[1]
redf=pd.DataFrame(index=newdf.columns,columns=[cn])
datadf=newdf.loc[cn].astype(int)
tempdf=covid19.r_covid(datadf)['Q0.5']
redf.loc[:,cn]=tempdf
redf.to_csv('output/reff_estim/{}_reffestim.csv'.format(cn))

