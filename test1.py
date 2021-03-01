import easyocr
reader = easyocr.Reader(['ch_sim', 'en'])  # need to run only once to load model into memory
result = reader.readtext('bing/black+protest/Scrapper_19.jpg')

print(result)