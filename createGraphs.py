import subprocess


for p in [ 0.56, 0.58, 0.6, 0.62, 0.64, 0.68, 0.7, 0.72, 0.74 ]:
    subprocess.run(['./venv/bin/python', 'bimodal_test.py', str(p), "4"])
