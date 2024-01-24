import json

solutions = {
    "date_fim_ValidDateQuery" : "ValidDateQuery",
    "date_fim_DateQuery" : "DateQuery",
}

with open("solutions.json", "w") as f:
    json.dump(solutions, f)
    