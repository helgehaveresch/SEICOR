#%%
import subprocess
import numpy as np
#%%

path = r"C:\Users\hhave\Documents\Promotion\scripts\SEICOR\fiber_to_pixel_SEICOR.txt"
with open(path, "r") as f:
        lines = f.readlines()

    # Processing the data
vea_fiber = []
pix_to_fiber = []

for line in lines:
    parts = line.split()
    vea_fiber.append(float(parts[0]))
    pix_to_fiber.append([int(x) for x in parts[1].split(",")]) 

vea_fiber = np.array(vea_fiber)
pix_to_fiber = np.array(pix_to_fiber)

Pantilt_vea = 105.9
vea_fiber = Pantilt_vea + vea_fiber
#%%
for i in range(1, 42):
        string=f"{i:02d}"
        subprocess.run(r'P:\exe\ave_spec.exe P:\data\wedel\pc\APR2025\250404ID.PC{} P:\data\wedel\pc\APR2025\250404_D.AV{} 60 /l 180'.format(string, string))
        subprocess.run(r'P:\exe\ave_spec.exe P:\data\wedel\pc\APR2025\250404ID.PC{} P:\data\wedel\pc\APR2025\250404ID.AV{} 5 /l {}'.format(string, string,vea_fiber[i-1]))
    
# %%
"""
for i in range(int(day_s), int(day_e)+1):
        filename = f"{year_and_month}{i:02d}_"
        print(f'Processing {filename}H.IMG')
        subprocess.run(r'P:\exe_64\resolut.exe {}D.OR?? /p=P:\data\pars\resolut.D'.format(filename))
   """ 
