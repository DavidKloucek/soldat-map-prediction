from helpers import create_model_file_path
from model import SequenceLengthException, load_model

print("\n\n", "█" * 30, "\n\n")

server = '=AUS7RAL|12 EuroShots #2'
# server = '=AUS7RAL|12 EuroShots'
model = load_model(create_model_file_path(server), "cpu")

['ctf_Viet', 'ctf_Ash', 'ctf_Cobra', 'ctf_Division',
 'ctf_Equinox', 'ctf_Guardian', 'ctf_Kampf', 'ctf_Laos', 'ctf_Maya']


def predict(input: list[str], target: str):
    try:
        print()
        print(f"Input: {', '.join(input)}")
        # start = time.time()
        maps, percs = model.predict(seq=input, topk=2)
        i = 0
        success = False
        for name, perc in zip(maps, percs):
            if i == 0 and perc >= 0.7 and name == target:
                success = True
            print(f" -> {name} | {perc:.4f}")
            i += 1
        print(f"{'SUCCEED' if success else 'FAILED'}")
        # end = time.time()
        # print(f"\nInference time: {end-start:.2f}s\n")
    except SequenceLengthException as e:
        print(e)


predict(['ctf_Kampf'], 'ctf_Laos')
predict(['ctf_Kampf', 'ctf_Laos'], 'ctf_Maya')
predict(['ctf_Division', 'ctf_Equinox'], 'ctf_Guardian')
predict(['ctf_Guardian', 'ctf_Kampf'], 'ctf_Laos')
predict(['ctf_Equinox', 'ctf_Guardian', 'ctf_Kampf'], 'ctf_Laos')
predict(['ctf_Division', 'ctf_Equinox', 'ctf_Guardian', 'ctf_Kampf'], 'ctf_Laos')
predict(['ctf_Cobra', 'ctf_Division', 'ctf_Equinox',
        'ctf_Guardian', 'ctf_Kampf'], 'ctf_Laos')
