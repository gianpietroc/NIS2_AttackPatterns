import pandas as pd
import re
import csv

capec = pd.read_csv("Capec.csv", index_col=False)
gt = pd.read_csv("", delimiter=";") #file of validation base
min_value = 4.5

gt = gt[gt['Validation'] > min_value]  # Visualizza il tipo di dato della colonna 'Total'
dir = "FinalExperimentsALL"
ss = pd.read_csv(dir + "/NormalSS/Alg3Article21(2models).csv") #file for the computed setting
#rag = pd.read_csv("RAG_Experiments/Measure 21/RAG_CAPEC_Exp_mitigates(21).csv")
rag = pd.read_csv(dir + "/ExtendedSS/phi4/Alg3Article21(2models).csv") #file for the computed setting

ss =  sorted(list(set(zip(ss['Measure'], ss['Attack Pattern']))))
rag =  sorted(list(set(zip(rag['Measure'], rag['Attack Pattern']))))
final_merge = sorted([value for value in ss if value in rag])
#final_merge = ss

print(f"Len SS: {len(ss)}")
print(f"Len PE: {len(rag)}")
print("Len Final intersection:", len(final_merge))
capec_pattern = re.compile(r'ChildOf:\s*CAPEC ID:\s*(\d+)', re.IGNORECASE)
related_map = {}

for _, row in capec.iterrows():
    child_id = int(row['ID'])  # The current CAPEC ID (child)
    related_text = row.get('Related Attack Patterns', '')
    
    # Ensure the related text is not NaN
    if pd.isna(related_text):
        related_text = ''
    
    # Find all parent IDs for this child
    parent_ids = sorted(list(map(int, capec_pattern.findall(str(related_text)))))
    
    # For each parent, add this child to the parent_to_children mapping
    for parent_id in parent_ids:
        if parent_id not in related_map:
            related_map[parent_id] = []
        related_map[parent_id].append(child_id)

#print(related_map)
expanded_pairs_exp = set()

for measure, attack_id in final_merge:
    expanded_pairs_exp.add((measure, attack_id))  # original
    related_ids = related_map.get(attack_id, [])
    for rel_id in related_ids:
        expanded_pairs_exp.add((measure, rel_id))

#expanded_pairs_exp = final_merge

expanded_pairs_gt = set()

for measure, attack_id in gt[['Measure', 'Attack Pattern']].apply(tuple, axis=1).tolist():
    expanded_pairs_gt.add((measure, attack_id))  # original
    related_ids = related_map.get(attack_id, [])
    for rel_id in related_ids:
        expanded_pairs_gt.add((measure, rel_id))

# Convert back to sorted list if needed
expanded_pairs_gt = sorted(expanded_pairs_gt)
gt = sorted(gt[['Measure', 'Attack Pattern']].apply(tuple, axis=1).tolist())

print(f"Len Final Extension Alg: {len(expanded_pairs_exp)}")
#print(f"GT {gt}")
print(f"Len Final Extension GT: {len(expanded_pairs_gt)}")

true_positives = [(measure, attack) for measure, attack in expanded_pairs_exp if (measure, attack) in expanded_pairs_gt]
false_positives = [(measure, attack) for measure, attack in expanded_pairs_exp if (measure, attack) not in expanded_pairs_gt]
false_negatives = [(measure, attack) for measure, attack in expanded_pairs_gt if (measure, attack) not in expanded_pairs_exp]

tp = len(true_positives)
fp = len(false_positives)
fn = len(false_negatives)

print("---")
print(f"TP: {tp:.2f}")
expanded_pairs_exp = final_merge

#print(f"FP: {fp:.2f}")
#print(f"FN: {fn:.2f}")

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

output_file_child = "Final_mapping_Article21_child.csv"

print("---")
true_positives = [(measure, attack) for measure, attack in expanded_pairs_exp if (measure, attack) in expanded_pairs_gt]
print(f"TP without child: {len(true_positives):.2f}")
tp = len(true_positives)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")


def extract_unique_elements(lst):
    elements = [pair[1] for pair in lst]
    return set(elements)

result = list(extract_unique_elements(expanded_pairs_gt))
#print(len(result))
print("---")
#print(true_positives)

# Filepath for the output CSV file
output_file_no_child = "Final_mapping_Article21_nochild.csv"

