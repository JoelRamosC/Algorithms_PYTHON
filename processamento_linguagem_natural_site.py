# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 16:38:51 2021

@author: joelr
"""

import nltk

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('portuguese')

import urllib.request
from bs4 import BeautifulSoup

response = urllib.request.urlopen('https://robo14bis.000webhostapp.com/')
html = response.read()

soup = BeautifulSoup(html,'html5lib')
text = soup.get_text(strip = True)
text = text.lower()


import re #realiza limpeza do texto

text = re.sub(r'[^\w\s]', ' ', text)
text = re.sub("\d+", ' ', text)


#extruturar palavras em lista
tokens = [t for t in text.split()]

clean_tokens = []

for token in tokens:
    if token not in stopwords and len(token) < 20:
        clean_tokens.append(token)
        
#frequencia de palavras
freq = nltk.FreqDist(clean_tokens)       

#vizualisar frequencia
for key,val in freq.items():
    print(str(key) + ':' + str(val))
 
freq.plot(20, cumulative=False)       