import pandas as pd
import json

df1=pd.read_csv("/mnt/remote/cowin/dp_cowin_standardised_anon.csv")
df2=pd.read_csv("/mnt/remote/icmr/dp_icmr_standardised_anon.csv")

cowin_count = df1.shape[0]
icmr_count = df2.shape[0]
# print("Cowin count:",cowin_count)
# print("ICMR count:", icmr_count)


common_values = set(df1['pk_mobno_hashed'].values).intersection(set(df2['pk_mobno_hashed'].values))
overlap_count = len(common_values)
# print(overlap_count)

overlap_percentage = (overlap_count/cowin_count)*100

overlap = {
    "count": overlap_count,
    "percentage": overlap_percentage
}
output_path = "/mnt/remote/output/overlap.json"

print("Writing training model to " + output_path)
with open(output_path, "w") as output_file:
    json.dump(overlap, output_file)