import json

class GetDataset:
    def __init__(self) -> None:
        pass
    
    def getDataset (self):
        with open('./dataset.json', 'r') as file:
            data = json.load(file)
            return data["data"]

    def  fomated_Data(self):
        data = self.getDataset()
        prices = []
        areas = []
        for item in data:
            if item["Area"] != "NaN":
                areas.append(item["Area"])
                prices.append(item["Price"])
            else:
                continue
        return [prices,areas]
#useage
# g = GetDataset()
# print(g.fomated_Data()[0])
# print(g.fomated_Data()[1])
# print(len(g.fomated_Data()[0]) == len(g.fomated_Data()[1]))
