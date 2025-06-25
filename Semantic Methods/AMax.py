import pandas as pd
import numpy as np

df = pd.read_csv('CSV\SS_ext_phi4(Measure7).csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.loc[:, ~df.columns.str.contains('^Jaccard')]
df = df.loc[:, ~df.columns.str.contains('^GPT')]

NUMBER_OF_MODELS = 4

models = ['MiniLM','ATTACK','Mpnet','Multilingual']

aps = list(set(df['Capec Id'])) #attack patterns

ap_meas = pd.DataFrame({"Attack Pattern": [], "Measure": [], "Max Value": [], "Models": []})

def rec_search(ap_rows, ap):
    global ap_meas

    max = 0
    m_list = []
    tmp = None

    for m in models:
        max_m = ap_rows.loc[ap_rows[m].idxmax()] #max from the different models
        max_m_value = max_m[m]

        if (max_m_value > max): 
                max = max_m_value
                max_m_meas = max_m['Measure']

        row = ap_rows.loc[ap_rows['Measure'] == max_m_meas]

    for m in models:
        if (row[m].values == max).any(): 
            m_list.append(m)

    print("m_list", m_list)
    if (len(m_list) == NUMBER_OF_MODELS - 1) :    
        print(max_m_meas)
        tmp = pd.DataFrame({"Attack Pattern": [ap], "Measure": [max_m_meas],
                                "Max Value": [max], "Models": [m_list]})
        ap_meas = pd.concat([ap_meas, tmp])

        i = ap_rows[ap_rows.Measure == max_m_meas].index
        ap_rows = ap_rows.drop(i)
        print(ap_rows)
        
        if not ap_rows.empty:
            rec_search(ap_rows, ap)
    
    else:
        return

for ap in aps:
    ap_rows = df.loc[df['Capec Id'] == ap]
    ap_rows = ap_rows.loc[:, ~ap_rows.columns.str.contains('^Capec')]
    ap_rows = ap_rows.reset_index()  # make sure indexes pair with number of rows

    rec_search(ap_rows, ap)

ap_meas['Attack Pattern'] = ap_meas['Attack Pattern'].replace(r'\.0$', '', regex=True).astype(np.int64)
ap_meas['Measure'] = ap_meas['Measure'].replace(r'\.0$', '', regex=True).astype(np.int64)

ap_meas.to_csv("FinalExperimentsALL/ExtendedSS/phi4/Alg0Article7(4models).csv", index=False, mode='w')
#ap_meas.to_csv("test.csv", index=False, mode='w')
