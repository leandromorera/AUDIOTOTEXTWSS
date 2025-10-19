import soundcard as sc

print("Output speakers (use these names for loopback):")
for spk in sc.all_speakers():
    print(" -", spk.name)

print("\nMicrophones (include_loopback=True may expose what-you-hear):")
for mic in sc.all_microphones(include_loopback=True):
    print(" -", mic.name)
