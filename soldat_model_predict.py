import time
from soldat_model import load_model

print("\n\n", "█" * 30, "\n\n")

model = load_model(path="soldat.pt", device="cpu")

x = ['ctf_Rotten', 'ctf_Snakebite', 'ctf_Steel', 'ctf_OldViet', 'ctf_Ash',
     'ctf_Cobra', 'ctf_Division', 'ctf_Equinox']

start = time.time()
maps, percs = model.predict(seq=x, topk=5)
end = time.time()

for name, perc in zip(maps, percs):
    print(f"{name} | {perc:.2f}")

print(f"\nInference time: {end-start:.2f}s\n")
