import sys

collected_txt = sys.argv[1]
uploaded_txt = sys.argv[2]

collected = open(collected_txt, "r").readlines()
uploaded = open(uploaded_txt, "r").readlines()

uploaded = [l.split("/")[0] for l in uploaded]
collected = [l.split("/")[1] for l in collected if "natural" not in l]

missing = set(collected).difference(set(uploaded))
print("\n".join(sorted(missing)))
print(len(missing))