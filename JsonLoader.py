import json

# f = open("gikAiJSONFile.json", "r", encoding='utf-8')
# data = f.read().replace('\n', '')
# jsonData = json.loads(data)
# print(jsonData["dishes"])

with open("gikAiJSONFile.json", "r", encoding='utf-8') as f:
    data = f.read()
    if data.startswith('\ufeff'):
        data = data[1:]
jsonData = json.loads(data)
print(jsonData["dishes"])