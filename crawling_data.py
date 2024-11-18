import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import json

load_dotenv() 

class Crawler:
    
    #check if it have data in dataset.json
    def __init__(self):
        self.check = self.check_size_dataset()    
    def check_size_dataset (self):
        file_path = './dataset.json'
        file_size = os.path.getsize(file_path)
        check = True if file_size > 0 else False
        return check
    
    #crawling data from url
    def crawl_data(self): 
        #range 1 - 50
        url = os.getenv("URL")
        header = {'User-Agent': os.getenv("User-Agent")}
        dataset_raw = []
        for i in range(1,51,1):
            res = requests.get(url=url+f"{i}",headers=header)
            if res.status_code == 200:
            
                html = res.text
                soup = BeautifulSoup(html,"html.parser")
            
                prices = soup.find_all("div",class_="inline-block text-secondary-base whitespace-nowrap font-bold text-3xl")

                uls = soup.find_all("ul",class_="list-none p-0 flex gap-x-5 font-semibold items-center")
                #print(len(prices) == len(uls))
                list_raw_data = self.getdata(uls=uls,prices=prices)
                dataset_raw += list_raw_data
            else:
                print("An error occurred:", res.status_code)
                
        return dataset_raw
    
    #get data from url 
    def getdata(self, uls, prices ):
        arr = []
        for ul, price in zip(uls, prices):
                lis = ul.find_all("li") 
                third_li = ""
                if len(lis) >= 3:  # Ensure there are at least 3 list items
                    if "m 2" in lis[2].get_text(" ", strip=True) and len(lis) == 4:
                        third_li = lis[2].get_text(" ", strip=True).replace(" m 2","") 
                    elif "m 2" in lis[1].get_text(" ", strip=True) and len(lis) == 3:
                        third_li = lis[1].get_text(" ", strip=True).replace(" m 2","")
                    else:
                        third_li = "NaN"
                else:
                    third_li = "NaN"#basecase
                raw_price = price.get_text(strip=True).replace("\u20ab ","")
                formated_price = raw_price.replace(" billion","") if(" billion" in raw_price) else (int(raw_price.replace(",",""))/1000000000)
                arr.append({"Area": str(third_li.replace(",","")), "Price": str(formated_price)})
        return arr
    
    #save to .json
    def save_dataset (self):
        if(self.check == False):
            a = self.crawl_data()
            datas = {
                "Units of Area":"m2",
                "Currency Units": "billion VND",
                "data":a
                }
            try:
                # Assuming `datas` is a dictionary or list and needs to be serialized to JSON
                with open("./dataset.json", "w") as dataset:
                    if isinstance(datas, (dict, list)):  # Check if datas is JSON serializable
                        json.dump(datas, dataset, indent=4)
                    else:
                        dataset.write(datas)  # Write raw string data if it's not JSON
                print("Saved to file")
            except (TypeError, IOError) as e:  # Handle specific exceptions
                print(f"Failed to save the dataset: {e}")
        else:
            print("Already had dataset!!")

#useage
# c = Crawler()
# c.save_dataset()
