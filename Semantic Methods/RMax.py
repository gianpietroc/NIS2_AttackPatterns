import pandas as pd
import numpy as np
from collections import Counter

df = pd.read_csv('CSV\SS_ext_phi4(Measure7).csv')

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.loc[:, ~df.columns.str.contains('^Jaccard')]
#df = df.loc[:, ~df.columns.str.contains('^GPT')]

models = ['MiniLM','ATTACK','Mpnet','Multilingual']
aps = list(set(df['Capec Id'])) #attack patterns

NUMBER_OF_MODELS = len(models)

m_list_tot = {}

ap_meas = pd.DataFrame({"Attack Pattern": [], "Measure": []})

m_list_values = []

def rec_search(ap_rows, ap):
    global ap_meas
    global m_list_values

    m_list_values.clear()
    m_list = []
     #list of measures for interesection
    tmp = None
    #print("---")
    for m in models:
        max_m = ap_rows.loc[ap_rows[m].idxmax()] #max from the different models
        max_m_value = max_m[m]
        max_m_meas = max_m['Measure']
        #print(ap, m, max_m_value, max_m_meas)
        
        m_list.append(max_m_meas)
        m_list_values.append(max_m_value)
        
    counts = Counter(m_list)
    
    occurrence_dict = dict(counts)
    max_item = max(occurrence_dict.items(), key=lambda item: item[1])

    max_value = max_item[1] #length
    #m_list = list(set(m_list))
    print("Max value:", max_item[0])
    count_threshold = 0

    print(m_list)

    if ((max_value >= NUMBER_OF_MODELS - 1)):# and count_threshold == max_value):   len(m_list) == 1
        #print(max_m_meas, m_list)
        tmp = pd.DataFrame({"Attack Pattern": [ap], "Measure": [int(max_item[0])],
        "MiniLM": [m_list_values[0]], "ATTACK": [m_list_values[1]],
                             "Mpnet": [m_list_values[2]],
                                "Multilingual": [m_list_values[3]]})
        
        ap_meas = pd.concat([ap_meas, tmp])
        m_list_tot[ap] = m_list[0]

        i = ap_rows[ap_rows.Measure == int(max_item[0])].index
        ap_rows = ap_rows.drop(i)
        print(ap_rows)
        
        if not ap_rows.empty:
            rec_search(ap_rows, ap)

    else:
        return 

for ap in aps:
    ap_rows = df[df['Capec Id'] == ap]
    #ap_rows = ap_rows.loc[:, ~ap_rows.columns.str.contains('^Capec')]
    #print("AP:", ap, "\n", ap_rows)
    rec_search(ap_rows, ap)

ap_meas['Attack Pattern'] = ap_meas['Attack Pattern'].replace(r'\.0$', '', regex=True).astype(np.int64)
ap_meas['Measure'] = ap_meas['Measure'].replace(r'\.0$', '', regex=True).astype(np.int64)

#print("AP + Measures:\n", ap_meas)
#print("AP + Measures:", m_list_tot)
print("Len", len(m_list_tot))

ap_meas.to_csv("FinalExperimentsALL/ExtendedSS/phi4/Alg3Article7(3models).csv", index=False, mode='w')
#ap_meas.to_csv("test2.csv", index=False, mode='w')
