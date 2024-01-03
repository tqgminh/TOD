import json

with open("train/dialogues_001.json") as f:
    dialogues = json.load(f)

all_data = []

for dialogue in dialogues:
    data = dict()
    data["dialogue_id"] = dialogue["dialogue_id"]
    data["turns"] = []
    turns = dialogue["turns"]
    for turn in turns:
        actions = turn["frames"][0]["actions"]
        tmp_turn = dict()
        tmp_turn["speaker"] = turn["speaker"]
        tmp_turn["actions"] = []
        for action in actions:
            tmp_action = dict()
            tmp_action["act"] = action["act"]
            tmp_action["slot"] = action["slot"]
            tmp_turn["actions"].append(tmp_action)
        data["turns"].append(tmp_turn)
    
    all_data.append(data)

with open('dialogue_data1.json', 'w') as f:
    json.dump(all_data, f, indent=4)
            
