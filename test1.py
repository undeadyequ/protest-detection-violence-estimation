#import easyocr
#reader = easyocr.Reader(['ch_sim', 'en'])  # need to run only once to load model into memory
#result = reader.readtext('bing/black+protest/Scrapper_19.jpg')

#print(result)

import pandas as pd



df = pd.read_csv("ucla_test.txt", delimiter=",")
sorted_df = df.sort_values("fname")

sorted_df.to_csv("annot_bfts_test.txt", sep=",", index=False)