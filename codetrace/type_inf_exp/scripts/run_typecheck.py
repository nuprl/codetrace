import subprocess
import glob
import sys
import os
d = sys.argv[1]

outs = []
for tsfile in glob.glob(f"{d}/*.ts"):
    tsfile = tsfile.split("/")[-1]
    print(tsfile)
    try:
        out = subprocess.run(
            ["npx", "--cache", str(os.environ["NPM_PACKAGES"]), "ts-node", "--typeCheck", tsfile],
            cwd=d,
            capture_output=True,
            timeout=120,
            text=True,
        ).stderr
        outs.append(out)
    except Exception as e:
        print("Error running tsnode: ", e)

outs = "\n".join(outs)
print(outs)