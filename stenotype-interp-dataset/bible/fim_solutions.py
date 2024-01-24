import json

solutions = {
    "bible_fim_unknown" : "unknown",
    "bible_fim_BookTitle" : "BookTitle",
}

with open("solutions.json", "w") as f:
    json.dump(solutions, f)
    