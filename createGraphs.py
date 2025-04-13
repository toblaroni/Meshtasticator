import subprocess


for p in [ 0.6, 0.65, 0.72 ]:
    print(f"Now running simulation for p = {p}")
    subprocess.run(['./venv/bin/python', 'bimodal_test.py', str(p), "4"])
