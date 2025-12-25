#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import unicodedata
import emoji
import pandas as pd
from unidecode import unidecode
import ftfy
from cleantext import clean
import nltk
from tqdm import tqdm
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


# In[ ]:


tqdm.pandas()


# In[2]:


df = pd.read_csv('comments_from_scraping.csv')


# In[3]:


comments = df['comment_text']


# In[4]:


factory = StopWordRemoverFactory()
stopword_remover = factory.create_stop_word_remover()

factory_stemmer = StemmerFactory()
stemmer = factory_stemmer.create_stemmer()


# In[5]:


extended_char_map = {
    # Latin Extended characters
    'ä': 'a', 'Ä': 'a', 'å': 'a', 'Å': 'a', 'æ': 'ae', 'Æ': 'ae',
    'ç': 'c', 'Ç': 'c', 'ð': 'd', 'Ð': 'd', 'ë': 'e', 'Ë': 'e',
    'ï': 'i', 'Ï': 'i', 'ñ': 'n', 'Ñ': 'n', 'ö': 'o', 'Ö': 'o',
    'ø': 'o', 'Ø': 'o', 'ü': 'u', 'Ü': 'u', 'ÿ': 'y', 'Ÿ': 'y',
    'ž': 'z', 'Ž': 'z', 'š': 's', 'Š': 's', 'č': 'c', 'Č': 'c',
    'ć': 'c', 'Ć': 'c', 'ğ': 'g', 'Ğ': 'g', 'ş': 's', 'Ş': 's',
    'ı': 'i', 'İ': 'i',
    
    # Greek letters yang sering digunakan sebagai pengganti
    'α': 'a', 'β': 'b', 'γ': 'g', 'δ': 'd', 'ε': 'e', 'ζ': 'z',
    'η': 'h', 'θ': 'th', 'ι': 'i', 'κ': 'k', 'λ': 'l', 'μ': 'm',
    'ν': 'n', 'ξ': 'x', 'ο': 'o', 'π': 'p', 'ρ': 'r', 'σ': 's',
    'τ': 't', 'υ': 'u', 'φ': 'ph', 'χ': 'ch', 'ψ': 'ps', 'ω': 'w',
    'Α': 'a', 'Β': 'b', 'Γ': 'g', 'Δ': 'd', 'Ε': 'e', 'Ζ': 'z',
    'Η': 'h', 'Θ': 'th', 'Ι': 'i', 'Κ': 'k', 'Λ': 'l', 'Μ': 'm',
    'Ν': 'n', 'Ξ': 'x', 'Ο': 'o', 'Π': 'p', 'Ρ': 'r', 'Σ': 's',
    'Τ': 't', 'Υ': 'u', 'Φ': 'ph', 'Χ': 'ch', 'Ψ': 'ps', 'Ω': 'w',
    
    # Cyrillic characters yang sering digunakan
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e',
    'ё': 'e', 'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k',
    'л': 'l', 'м': 'm', 'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r',
    'с': 's', 'т': 't', 'у': 'u', 'ф': 'f', 'х': 'h', 'ц': 'ts',
    'ч': 'ch', 'ш': 'sh', 'щ': 'sch', 'ъ': '', 'ы': 'y', 'ь': '',
    'э': 'e', 'ю': 'yu', 'я': 'ya',
    'А': 'a', 'Б': 'b', 'В': 'v', 'Г': 'g', 'Д': 'd', 'Е': 'e',
    'Ё': 'e', 'Ж': 'zh', 'З': 'z', 'И': 'i', 'Й': 'y', 'К': 'k',
    'Л': 'l', 'М': 'm', 'Н': 'n', 'О': 'o', 'П': 'p', 'Р': 'r',
    'С': 's', 'Т': 't', 'У': 'u', 'Ф': 'f', 'Х': 'h', 'Ц': 'ts',
    'Ч': 'ch', 'Ш': 'sh', 'Щ': 'sch', 'Ъ': '', 'Ы': 'y', 'Ь': '',
    'Э': 'e', 'Ю': 'yu', 'Я': 'ya',
    
    # Mathematical alphanumeric symbols
    '𝐀': 'a', '𝐁': 'b', '𝐂': 'c', '𝐃': 'd', '𝐄': 'e', '𝐅': 'f',
    '𝐆': 'g', '𝐇': 'h', '𝐈': 'i', '𝐉': 'j', '𝐊': 'k', '𝐋': 'l',
    '𝐌': 'm', '𝐍': 'n', '𝐎': 'o', '𝐏': 'p', '𝐐': 'q', '𝐑': 'r',
    '𝐒': 's', '𝐓': 't', '𝐔': 'u', '𝐕': 'v', '𝐖': 'w', '𝐗': 'x',
    '𝐘': 'y', '𝐙': 'z', '𝐚': 'a', '𝐛': 'b', '𝐜': 'c', '𝐝': 'd',
    '𝐞': 'e', '𝐟': 'f', '𝐠': 'g', '𝐡': 'h', '𝐢': 'i', '𝐣': 'j',
    '𝐤': 'k', '𝐥': 'l', '𝐦': 'm', '𝐧': 'n', '𝐨': 'o', '𝐩': 'p',
    '𝐪': 'q', '𝐫': 'r', '𝐬': 's', '𝐭': 't', '𝐮': 'u', '𝐯': 'v',
    '𝐰': 'w', '𝐱': 'x', '𝐲': 'y', '𝐳': 'z',
    
    '𝐴': 'a', '𝐵': 'b', '𝐶': 'c', '𝐷': 'd', '𝐸': 'e', '𝐹': 'f',
    '𝐺': 'g', '𝐻': 'h', '𝐼': 'i', '𝐽': 'j', '𝐾': 'k', '𝐿': 'l',
    '𝑀': 'm', '𝑁': 'n', '𝑂': 'o', '𝑃': 'p', '𝑄': 'q', '𝑅': 'r',
    '𝑆': 's', '𝑇': 't', '𝑈': 'u', '𝑉': 'v', '𝑊': 'w', '𝑋': 'x',
    '𝑌': 'y', '𝑍': 'z', '𝑎': 'a', '𝑏': 'b', '𝑐': 'c', '𝑑': 'd',
    '𝑒': 'e', '𝑓': 'f', '𝑔': 'g', 'ℎ': 'h', '𝑖': 'i', '𝑗': 'j',
    '𝑘': 'k', '𝑙': 'l', '𝑚': 'm', '𝑛': 'n', '𝑜': 'o', '𝑝': 'p',
    '𝑞': 'q', '𝑟': 'r', '𝑠': 's', '𝑡': 't', '𝑢': 'u', '𝑣': 'v',
    '𝑤': 'w', '𝑥': 'x', '𝑦': 'y', '𝑧': 'z',
    
    '𝒜': 'a', 'ℬ': 'b', '𝒞': 'c', '𝒟': 'd', 'ℰ': 'e', 'ℱ': 'f',
    '𝒢': 'g', 'ℋ': 'h', 'ℐ': 'i', '𝒥': 'j', '𝒦': 'k', 'ℒ': 'l',
    'ℳ': 'm', '𝒩': 'n', '𝒪': 'o', '𝒫': 'p', '𝒬': 'q', 'ℛ': 'r',
    '𝒮': 's', '𝒯': 't', '𝒰': 'u', '𝒱': 'v', '𝒲': 'w', '𝒳': 'x',
    '𝒴': 'y', '𝒵': 'z', '𝒶': 'a', '𝒷': 'b', '𝒸': 'c', '𝒹': 'd',
    'ℯ': 'e', '𝒻': 'f', 'ℊ': 'g', '𝒽': 'h', '𝒾': 'i', '𝒿': 'j',
    '𝓀': 'k', '𝓁': 'l', '𝓂': 'm', '𝓃': 'n', 'ℴ': 'o', '𝓅': 'p',
    '𝓆': 'q', '𝓇': 'r', '𝓈': 's', '𝓉': 't', '𝓊': 'u', '𝓋': 'v',
    '𝓌': 'w', '𝓍': 'x', '𝓎': 'y', '𝓏': 'z',
    
    '𝓐': 'a', '𝓑': 'b', '𝓒': 'c', '𝓓': 'd', '𝓔': 'e', '𝓕': 'f',
    '𝓖': 'g', '𝓗': 'h', '𝓘': 'i', '𝓙': 'j', '𝓚': 'k', '𝓛': 'l',
    '𝓜': 'm', '𝓝': 'n', '𝓞': 'o', '𝓟': 'p', '𝓠': 'q', '𝓡': 'r',
    '𝓢': 's', '𝓣': 't', '𝓤': 'u', '𝓥': 'v', '𝓦': 'w', '𝓧': 'x',
    '𝓨': 'y', '𝓩': 'z', '𝓪': 'a', '𝓫': 'b', '𝓬': 'c', '𝓭': 'd',
    '𝓮': 'e', '𝓯': 'f', '𝓰': 'g', '𝓱': 'h', '𝓲': 'i', '𝓳': 'j',
    '𝓴': 'k', '𝓵': 'l', '𝓶': 'm', '𝓷': 'n', '𝓸': 'o', '𝓹': 'p',
    '𝓺': 'q', '𝓻': 'r', '𝓼': 's', '𝓽': 't', '𝓾': 'u', '𝓿': 'v',
    '𝔀': 'w', '𝔁': 'x', '𝔂': 'y', '𝔃': 'z',
    
    '𝔄': 'a', '𝔅': 'b', 'ℭ': 'c', '𝔇': 'd', '𝔈': 'e', '𝔉': 'f',
    '𝔊': 'g', 'ℌ': 'h', 'ℑ': 'i', '𝔍': 'j', '𝔎': 'k', '𝔏': 'l',
    '𝔐': 'm', '𝔑': 'n', '𝔒': 'o', '𝔓': 'p', '𝔔': 'q', 'ℜ': 'r',
    '𝔖': 's', '𝔗': 't', '𝔘': 'u', '𝔙': 'v', '𝔚': 'w', '𝔛': 'x',
    '𝔜': 'y', 'ℨ': 'z', '𝔞': 'a', '𝔟': 'b', '𝔠': 'c', '𝔡': 'd',
    '𝔢': 'e', '𝔣': 'f', '𝔤': 'g', '𝔥': 'h', '𝔦': 'i', '𝔧': 'j',
    '𝔨': 'k', '𝔩': 'l', '𝔪': 'm', '𝔫': 'n', '𝔬': 'o', '𝔭': 'p',
    '𝔮': 'q', '𝔯': 'r', '𝔰': 's', '𝔱': 't', '𝔲': 'u', '𝔳': 'v',
    '𝔴': 'w', '𝔵': 'x', '𝔶': 'y', '𝔷': 'z',
    
    '𝕬': 'a', '𝕭': 'b', '𝕮': 'c', '𝕯': 'd', '𝕰': 'e', '𝕱': 'f',
    '𝕲': 'g', '𝕳': 'h', '𝕴': 'i', '𝕵': 'j', '𝕶': 'k', '𝕷': 'l',
    '𝕸': 'm', '𝕹': 'n', '𝕺': 'o', '𝕻': 'p', '𝕼': 'q', '𝕽': 'r',
    '𝕾': 's', '𝕿': 't', '𝖀': 'u', '𝖁': 'v', '𝖂': 'w', '𝖃': 'x',
    '𝖄': 'y', '𝖅': 'z', '𝖆': 'a', '𝖇': 'b', '𝖈': 'c', '𝖉': 'd',
    '𝖊': 'e', '𝖋': 'f', '𝖌': 'g', '𝖍': 'h', '𝖎': 'i', '𝖏': 'j',
    '𝖐': 'k', '𝖑': 'l', '𝖒': 'm', '𝖓': 'n', '𝖔': 'o', '𝖕': 'p',
    '𝖖': 'q', '𝖗': 'r', '𝖘': 's', '𝖙': 't', '𝖚': 'u', '𝖛': 'v',
    '𝖜': 'w', '𝖝': 'x', '𝖞': 'y', '𝖟': 'z',
    
    '𝖠': 'a', '𝖡': 'b', '𝖢': 'c', '𝖣': 'd', '𝖤': 'e', '𝖥': 'f',
    '𝖦': 'g', '𝖧': 'h', '𝖨': 'i', '𝖩': 'j', '𝖪': 'k', '𝖫': 'l',
    '𝖬': 'm', '𝖭': 'n', '𝖮': 'o', '𝖯': 'p', '𝖰': 'q', '𝖱': 'r',
    '𝖲': 's', '𝖳': 't', '𝖴': 'u', '𝖵': 'v', '𝖶': 'w', '𝖷': 'x',
    '𝖸': 'y', '𝖹': 'z', '𝖺': 'a', '𝖻': 'b', '𝖼': 'c', '𝖽': 'd',
    '𝖾': 'e', '𝖿': 'f', '𝗀': 'g', '𝗁': 'h', '𝗂': 'i', '𝗃': 'j',
    '𝗄': 'k', '𝗅': 'l', '𝗆': 'm', '𝗇': 'n', '𝗈': 'o', '𝗉': 'p',
    '𝗊': 'q', '𝗋': 'r', '𝗌': 's', '𝗍': 't', '𝗎': 'u', '𝗏': 'v',
    '𝗐': 'w', '𝗑': 'x', '𝗒': 'y', '𝗓': 'z',
    
    '𝗔': 'a', '𝗕': 'b', '𝗖': 'c', '𝗗': 'd', '𝗘': 'e', '𝗙': 'f',
    '𝗚': 'g', '𝗛': 'h', '𝗜': 'i', '𝗝': 'j', '𝗞': 'k', '𝗟': 'l',
    '𝗠': 'm', '𝗡': 'n', '𝗢': 'o', '𝗣': 'p', '𝗤': 'q', '𝗥': 'r',
    '𝗦': 's', '𝗧': 't', '𝗨': 'u', '𝗩': 'v', '𝗪': 'w', '𝗫': 'x',
    '𝗬': 'y', '𝗭': 'z', '𝗮': 'a', '𝗯': 'b', '𝗰': 'c', '𝗱': 'd',
    '𝗲': 'e', '𝗳': 'f', '𝗴': 'g', '𝗵': 'h', '𝗶': 'i', '𝗷': 'j',
    '𝗸': 'k', '𝗹': 'l', '𝗺': 'm', '𝗻': 'n', '𝗼': 'o', '𝗽': 'p',
    '𝗾': 'q', '𝗿': 'r', '𝘀': 's', '𝘁': 't', '𝘂': 'u', '𝘃': 'v',
    '𝘄': 'w', '𝘅': 'x', '𝘆': 'y', '𝘇': 'z',
    
    '𝘈': 'a', '𝘉': 'b', '𝘊': 'c', '𝘋': 'd', '𝘌': 'e', '𝘍': 'f',
    '𝘎': 'g', '𝘏': 'h', '𝘐': 'i', '𝘑': 'j', '𝘒': 'k', '𝘓': 'l',
    '𝘔': 'm', '𝘕': 'n', '𝘖': 'o', '𝘗': 'p', '𝘘': 'q', '𝘙': 'r',
    '𝘚': 's', '𝘛': 't', '𝘜': 'u', '𝘝': 'v', '𝘞': 'w', '𝘟': 'x',
    '𝘠': 'y', '𝘡': 'z', '𝘢': 'a', '𝘣': 'b', '𝘤': 'c', '𝘥': 'd',
    '𝘦': 'e', '𝘧': 'f', '𝘨': 'g', '𝘩': 'h', '𝘪': 'i', '𝘫': 'j',
    '𝘬': 'k', '𝘭': 'l', '𝘮': 'm', '𝘯': 'n', '𝘰': 'o', '𝘱': 'p',
    '𝘲': 'q', '𝘳': 'r', '𝘴': 's', '𝘵': 't', '𝘶': 'u', '𝘷': 'v',
    '𝘸': 'w', '𝘹': 'x', '𝘺': 'y', '𝘻': 'z',
    
    '𝘼': 'a', '𝘽': 'b', '𝘾': 'c', '𝘿': 'd', '𝙀': 'e', '𝙁': 'f',
    '𝙂': 'g', '𝙃': 'h', '𝙄': 'i', '𝙅': 'j', '𝙆': 'k', '𝙇': 'l',
    '𝙈': 'm', '𝙉': 'n', '𝙊': 'o', '𝙋': 'p', '𝙌': 'q', '𝙍': 'r',
    '𝙎': 's', '𝙏': 't', '𝙐': 'u', '𝙑': 'v', '𝙒': 'w', '𝙓': 'x',
    '𝙔': 'y', '𝙕': 'z', '𝙖': 'a', '𝙗': 'b', '𝙘': 'c', '𝙙': 'd',
    '𝙚': 'e', '𝙛': 'f', '𝙜': 'g', '𝙝': 'h', '𝙞': 'i', '𝙟': 'j',
    '𝙠': 'k', '𝙡': 'l', '𝙢': 'm', '𝙣': 'n', '𝙤': 'o', '𝙥': 'p',
    '𝙦': 'q', '𝙧': 'r', '𝙨': 's', '𝙩': 't', '𝙪': 'u', '𝙫': 'v',
    '𝙬': 'w', '𝙭': 'x', '𝙮': 'y', '𝙯': 'z',
    
    '𝙰': 'a', '𝙱': 'b', '𝙲': 'c', '𝙳': 'd', '𝙴': 'e', '𝙵': 'f',
    '𝙶': 'g', '𝙷': 'h', '𝙸': 'i', '𝙹': 'j', '𝙺': 'k', '𝙻': 'l',
    '𝙼': 'm', '𝙽': 'n', '𝙾': 'o', '𝙿': 'p', '𝚀': 'q', '𝚁': 'r',
    '𝚂': 's', '𝚃': 't', '𝚄': 'u', '𝚅': 'v', '𝚆': 'w', '𝚇': 'x',
    '𝚈': 'y', '𝚉': 'z', '𝚊': 'a', '𝚋': 'b', '𝚌': 'c', '𝚍': 'd',
    '𝚎': 'e', '𝚏': 'f', '𝚐': 'g', '𝚑': 'h', '𝚒': 'i', '𝚓': 'j',
    '𝚔': 'k', '𝚕': 'l', '𝚖': 'm', '𝚗': 'n', '𝚘': 'o', '𝚙': 'p',
    '𝚚': 'q', '𝚛': 'r', '𝚜': 's', '𝚝': 't', '𝚞': 'u', '𝚟': 'v',
    '𝚠': 'w', '𝚡': 'x', '𝚢': 'y', '𝚣': 'z',
    
    # Special symbols and brackets
    '【': ' ', '】': ' ', '『': ' ', '』': ' ', '〖': ' ', '〗': ' ',
    '「': ' ', '」': ' ', '｢': ' ', '｣': ' ', '〔': ' ', '〕': ' ',
    '〈': ' ', '〉': ' ', '《': ' ', '》': ' ', '«': ' ', '»': ' ',
    '〝': ' ', '〞': ' ', '＂': ' ', '‟': ' ', '〟': ' ',
    '：': ' ', '；': ' ', '，': ' ', '。': ' ', '、': ' ',
    '！': ' ', '？': ' ', '～': ' ', '‧': ' ', '・': ' ',
    '¢': ' ', '@': ' ', '®': ' ', '©': ' ', '™': ' ', '?': ' ',
    '♜': ' ', '☆': ' ', '🎯': ' ', '🐟': ' ', '❈': ' ', '✷': ' ',
    '🎀': ' ', '💮': 'o', '🏵': 'o', '|': ' ', '!': ' ', '¤': ' ',
    '*': ' ', "'": ' ', '~': ' ', '`': ' ', '¯': ' ', '•': ' ', 
    ',': ' ', '¸': ' ', '´': ' ', 'Δ': 'a', 'ᗯ': 'w', 'ᗩ': 'a',
    '†': 't', '丅': 't', 'Ⓞ': 'o', '~': ' ', '`': ' ', '´': ' ',
    
    # TAMBAHAN BARU UNTUK PERBAIKAN ARWANATOTO:
    # Greek and special characters untuk "arwanatoto"
    'Ř': 'r', 'ά': 'a', 
    'ǟ': 'a', 'ʀ': 'r', 'ա': 'w', 'ռ': 'n', 'ȶ': 't', 'օ': 'o',
    'ñ': 'n', 
    'A҉': 'a', 'R҉': 'r', 'W҉': 'w', 'N҉': 'n', 'T҉': 't', 'O҉': 'o',
    
    # Special decorated characters
    '𝒜': 'a', '𝑅': 'r', '𝒲': 'w', '𝒜': 'a', '𝒩': 'n', '𝒯': 't', 
    '💮': 'o', '🏵': 'o', '🍬': 'o', '♡': 'o', '💞': 'o',
    
    # Mathematical symbols
    '𝐀': 'a', '𝐑': 'r', '𝐖': 'w', '𝐀': 'a', '𝐍': 'n', '𝐓': 't', '𝐎': 'o',
    '𝓐': 'a', '𝓡': 'r', '𝓦': 'w', '𝓐': 'a', '𝓝': 'n', '𝓣': 't', '𝓞': 'o',
    '𝔄': 'a', 'ℜ': 'r', '𝔚': 'w', '𝔄': 'a', '𝔑': 'n', '𝔗': 't', '𝔒': 'o',
    
    # Tambahkan lebih banyak variant
    '🅐': 'a', '🅡': 'r', '🅦': 'w', '🅝': 'n', '🅣': 't', '🅞': 'o',
    'Ⓐ': 'a', 'Ⓡ': 'r', 'Ⓦ': 'w', 'Ⓝ': 'n', 'Ⓣ': 't', 'Ⓞ': 'o',
    
    # Special case characters
    'σ': 'o', '𝓽': 't', '𝐎': 'o', '𝕒': 'a', 'т': 't', 'ή': 'n',
    
    # Emoji dan simbol khusus
    '🥇': '1', '🏆': ' trophy ', '🎯': ' target ', '💎': ' diamond ',
    '💰': ' money ', '💸': ' money ', '🤑': ' money ', '💵': ' money ',
    '💴': ' money ', '💶': ' money ', '💷': ' money ', '💳': ' card ',
    '💹': ' chart ', '↗': ' up ', '⬆': ' up ', '↘': ' down ', 
    '⬇': ' down ', '⬅': ' left ', '➡': ' right ', '↔': ' both ',
    '🔝': ' top ', '🔙': ' back ', '🔛': ' on ', '🔜': ' soon ',
    '🔚': ' end ', '✅': ' yes ', '✔': ' yes ', '✓': ' yes ',
    '❌': ' no ', '✖': ' no ', '❎': ' no ', '⚠': ' warning ',
}

emoji_letters = {
    '🇦': 'a', '🇧': 'b', '🇨': 'c', '🇩': 'd', '🇪': 'e',
    '🇫': 'f', '🇬': 'g', '🇭': 'h', '🇮': 'i', '🇯': 'j',
    '🇰': 'k', '🇱': 'l', '🇲': 'm', '🇳': 'n', '🇴': 'o',
    '🇵': 'p', '🇶': 'q', '🇷': 'r', '🇸': 's', '🇹': 't',
    '🇺': 'u', '🇻': 'v', '🇼': 'w', '🇽': 'x', '🇾': 'y',
    '🇿': 'z', '🅰': 'a', '🅱': 'b', '🅲': 'c', '🅳': 'd',
    '🅴': 'e', '🅵': 'f', '🅶': 'g', '🅷': 'h', '🅸': 'i',
    '🅹': 'j', '🅺': 'k', '🅻': 'l', '🅼': 'm', '🅽': 'n',
    '🅾': 'o', '🅿': 'p', '🆀': 'q', '🆁': 'r', '🆂': 's',
    '🆃': 't', '🆄': 'u', '🆅': 'v', '🆆': 'w', '🆇': 'x',
    '🆈': 'y', '🆉': 'z', '🅐': 'a', '🅑': 'b', '🅒': 'c',
    '🅓': 'd', '🅔': 'e', '🅕': 'f', '🅖': 'g', '🅗': 'h',
    '🅘': 'i', '🅙': 'j', '🅚': 'k', '🅛': 'l', '🅜': 'm',
    '🅝': 'n', '🅞': 'o', '🅟': 'p', '🅠': 'q', '🅡': 'r',
    '🅢': 's', '🅣': 't', '🅤': 'u', '🅥': 'v', '🅦': 'w',
    '🅧': 'x', '🅨': 'y', '🅩': 'z', 'Ⓐ': 'a', 'Ⓑ': 'b',
    'Ⓒ': 'c', 'Ⓓ': 'd', 'Ⓔ': 'e', 'Ⓕ': 'f', 'Ⓖ': 'g',
    'Ⓗ': 'h', 'Ⓘ': 'i', 'Ⓙ': 'j', 'Ⓚ': 'k', 'Ⓛ': 'l',
    'Ⓜ': 'm', 'Ⓝ': 'n', 'Ⓞ': 'o', 'Ⓟ': 'p', 'Ⓠ': 'q',
    'Ⓡ': 'r', 'Ⓢ': 's', 'Ⓣ': 't', 'Ⓤ': 'u', 'Ⓥ': 'v',
    'Ⓦ': 'w', 'Ⓧ': 'x', 'Ⓨ': 'y', 'Ⓩ': 'z',
}

emoji_numbers = {
    '1️⃣': '1', '2️⃣': '2', '3️⃣': '3', '4️⃣': '4', '5️⃣': '5', 
    '6️⃣': '6', '7️⃣': '7', '8️⃣': '8', '9️⃣': '9', '0️⃣': '0',
    '➀': '1', '➁': '2', '➂': '3', '➃': '4', '➄': '5',
    '➅': '6', '➆': '7', '➇': '8', '➈': '9', '🄋': '0',
    '🥇': '1', '🥈': '2', '🥉': '3', '🏆': ' trophy ', '🎯': ' target ',
    '❶': '1', '❷': '2', '❸': '3', '❹': '4', '❺': '5',
    '❻': '6', '❼': '7', '❽': '8', '❾': '9', '❿': '10',
}

number_map = {
    '0': 'o', '1': 'i', '2': 'z', '3': 'e', '4': 'a',
    '5': 's', '6': 'g', '7': 't', '8': 'b', '9': 'g',
    '!': 'i', '@': 'a', '$': 's', '+': 't'
}

indonesian_slang_dict = {
    'yg': 'yang',
    'gk': 'tidak',
    'gak': 'tidak',
    'ga': 'tidak',
    'jgn': 'jangan',
    'tdk': 'tidak',
    'nggak': 'tidak',
    'ngga': 'tidak',
    'dgn': 'dengan',
    'bg': 'bang',
    'banget': 'sekali',
    'bgt': 'sekali',
    'banget': 'sekali',
    'bngt': 'sekali',
    'sih': '',
    'dong': '',
    'deh': '',
    'lah': '',
    'nih': 'ini',
    'tuh': 'itu',
    'lu': 'kamu',
    'loe': 'kamu',
    'gw': 'saya',
    'gua': 'saya',
    'gue': 'saya',
    'ane': 'saya',
    'ente': 'anda',
    'lo': 'kamu',
    'elu': 'kamu',
    'wkwk': 'haha',
    'wkwkwk': 'haha',
    'hehe': 'haha',
    'haha': '',
    'wkwkwkwk': 'haha',
    'anjir': 'astaga',
    'anjay': 'astaga',
    'cuy': '',
    'bro': '',
    'sob': '',
    'gan': '',
    'sis': '',
    'bang': '',
    'mas': '',
    'mbak': '',
    'pak': '',
    'bu': '',
    'om': '',
    'tante': '',
    'dek': '',
    'kak': ''
}

brand_map = {
    "GA RUDa HO KI": "garudahoki",
    "GA 𝐑𝐔𝐃a 𝐇𝐎 KI": "garudahoki",  
    "GA 𝐑𝐔𝐃a 𝐇𝐎 Ki": "garudahoki",
    "Ga ruda Hoki": "garudahoki",
    "Gar uda-Ho ki": "garudahoki",
    "Garuda Ho ki": "garudahoki",
    "P U L A U W I N": "pulauwin",
    "ρυℓαυωιɴ": "pulauwin",
    "PUL AUWIN": "pulauwin",
    "P͟͟U͟͟L͟͟A͟͟U͟͟ W͟͟I͟͟N͟͟": "pulauwin",
    "𝕊 𝕌 𝕂 𝕌 𝟠 𝟠": "suku88",
    "𝕊 𝕌 𝕂 𝕌 𝟠 𝟠🔥🔥🔥": "suku88",
    "N I C E": "nice",
    "T0GEL62": "togel62"
}

custom_stopwords = {
    'yang', 'di', 'ke', 'dari', 'pada', 'dalam', 'untuk', 'dengan', 
    'adalah', 'atau', 'tapi', 'dan', 'jika', 'karena', 'serta', 
    'oleh', 'itu', 'ini', 'saja', 'hanya', 'pun', 'lah', 'kah',
    'tah', 'pun', 'nya', 'ku', 'mu', 'kau', 'kami', 'kita', 'mereka',
    'saya', 'kamu', 'dia', 'beliau', 'para', 'si', 'sang', 'itu',
    'hal', 'per', 'oleh', 'agar', 'supaya', 'meski', 'walau',
    'sebab', 'karena', 'jika', 'kalau', 'apabila', 'seandainya',
    'agar', 'supaya', 'guna', 'untuk', 'demi', 'sebagai', 'laksana',
    'bak', 'ibarat', 'serupa', 'tanpa', 'dengan', 'secara', 'sambil',
    'seraya', 'selagi', 'sementara', 'ketika', 'tatkala', 'sewaktu',
    'sebelum', 'sesudah', 'setelah', 'hingga', 'sampai', 'semenjak',
    'sedari', 'seraya', 'sambil', 'seraya', 'sambil', 'seraya'
}


# In[6]:


def fix_word_spacing(text):
    cleaned_chars = []
    for char in text:
        # Pertahankan karakter ASCII yang printable
        if ord(char) < 128 and char.isprintable():
            cleaned_chars.append(char)
        # Untuk karakter dengan aksen, ambil base char-nya
        elif unicodedata.combining(char):
            continue  # skip combining characters (aksen, dll)
        # Untuk karakter khusus lainnya, ganti dengan spasi
        elif not char.isprintable() or ord(char) > 127:
            cleaned_chars.append(' ')
        else:
            cleaned_chars.append(char)
    
    cleaned_text = ''.join(cleaned_chars)
    
    # Step 3: Normalisasi spasi (tapi jangan terlalu agresif)
    cleaned_text = re.sub(r'[^\S\n]+', ' ', cleaned_text)  # ganti multiple spaces dengan satu space
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)  # pertahankan line breaks
    
    return cleaned_text.strip()


# In[7]:


def replace_unicode(text):
    cleaned_chars = []
    for char in text:
        # Peta karakter khusus
        if char in extended_char_map:
            cleaned_chars.append(extended_char_map[char])
            continue

        # Biarkan tanda baca umum dan dash/em dash
        if char in '.,!?;:()[]{}"\'-—–…':
            cleaned_chars.append(char)
            continue

        # Titik mirip jadi titik ASCII
        if char in ['․', '‧', '·', '•', '・', '｡', '。']:
            cleaned_chars.append('.')
            continue

        # Hilangkan diakritik
        decomposed = unicodedata.normalize('NFKD', char)
        base_char = ''.join(c for c in decomposed if not unicodedata.combining(c))

        # Pertahankan karakter printable (termasuk emoji)
        if base_char.isprintable():
            cleaned_chars.append(base_char)
        else:
            cleaned_chars.append(' ')
            
    return ''.join(cleaned_chars)


# In[8]:


def replace_emoji_number(text):
    for emo, num in emoji_numbers.items():
        text = text.replace(emo, f"<NUM>{num}</NUM>")
    
        # Lindungi tag sementara agar tidak ikut terhapus di regex cleaning
        text = text.replace("<NUM>", "§OPEN§").replace("</NUM>", "§CLOSE§")
        
        # Hilangkan simbol non-alfanumerik tanpa mengganggu huruf beraksen (udah dihapus di replace_unicode)
        text = re.sub(r'[^\w\s§.,!?;:\'\"-]', '', text)
        
        # Kembalikan tag
        text = text.replace("§OPEN§", "<NUM>").replace("§CLOSE§", "</NUM>")
        
        # Gabungkan angka dari emoji yang menempel dengan huruf
        text = re.sub(r'([a-zA-Z])<NUM>(\d+)</NUM>([a-zA-Z])', r'\1\2\3', text)
        text = re.sub(r'<NUM>(\d+)</NUM>', r'\1', text)
        
        # Rapikan spasi
        text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# In[9]:


def replace_emoji_letter(text):
    # Hapus zero-width chars
    text = re.sub(r'[\u200b\u200c\u200d\uFEFF]', '', text)

    # Tangani regional indicator flags
    def combine_flag(match):
        return ' ' + ''.join(emoji_letters.get(c, '') for c in match.group()) + ' '
    
    text = re.sub(r'([\U0001F1E6-\U0001F1FF]+)', combine_flag, text)

    return text


# In[10]:


def replace_brand(text):
    for key, val in brand_map.items():
        # case-insensitive
        text = re.sub(re.escape(key), val, text, flags=re.IGNORECASE)
    return text


# In[11]:


def remove_text_decorations(text):
    """Hapus dekorasi teks seperti |!¤*'~``~'*¤!| dan sejenisnya"""
    # Pattern untuk decorated text dengan berbagai simbol
    decoration_patterns = [
        r'[|!¤*\'~`¯,¸øº°∙▪■□▢▣▤▥▦▧▨▩▪▫▬▭▮▯▰▱▲△▴▵▶▷▸▹►▻▼▽▾▿◀◁◂◃◄◅◆◇◈◉◊○◌◍◎●◐◑◒◓◔◕◖◗◘◙◚◛◜◝◞◟◠◡◢◣◤◥◦◧◨◩◪◫◬◭◮◯◰◱◲◳◴◵◶◷◸◹◺◻◼◽◾◿]+',
    ]
    
    for pattern in decoration_patterns:
        text = re.sub(pattern, ' ', text)
    
    return text


# In[12]:


def clean_text_for_nlp(text: str):
    if not text or not isinstance(text, str):
        return ""
    
    text = replace_brand(text)
    # misal brand GA RUDa HO Ki → garuda hoki
    # text = re.sub(r'\b(' + replace_brand_dynamic(text) + r')\b', r' \1 ', text)

    
    # Fix encoding / unicode
    text = ftfy.fix_text(text)

    # 1️⃣ Gabungkan flag jadi huruf (🇦🇷 → ar)
    def combine_flags(match):
        return ''.join(emoji_letters.get(c, '') for c in match.group())
    text = re.sub(r'([\U0001F1E6-\U0001F1FF]+)', combine_flags, text)

    # 2️⃣ Tangani emoji angka (KYT4️⃣D → KYT4D)
    for emo, num in emoji_numbers.items():
        text = text.replace(emo, num)

    emoji_pattern = re.compile(
        r'['
        r'\U0001F300-\U0001F5FF'  # simbol
        r'\U0001F600-\U0001F64F'  # wajah
        r'\U0001F680-\U0001F6FF'  # transportasi
        r'\U0001F1E6-\U0001F1FF'  # regional indicator / bendera
        r'\u2764\uFE0F'           # ❤️
        r'\U0001F90E'              # 🤎
        r'✨🌟🔥😍💎🎰⚡'            # emoji populer tambahan
        r']+', flags=re.UNICODE
    )
    text = emoji_pattern.sub(' ', text)

    # 4️⃣ Hapus zero-width chars
    text = re.sub(r'[\u200b\u200c\u200d\uFEFF]', '', text)

    # 5️⃣ Rapikan spasi dan tanda baca
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'([.,!?;:])(?=\w)', r'\1 ', text)

    text = replace_unicode(text)
    text = replace_emoji_letter(text) 
    text = replace_emoji_number(text)  
    text = remove_text_decorations(text)

    text = re.sub(r'(?<=\w)\.(?=\w)', '. ', text)

    # 6️⃣ Tangani kasus khusus H.Malih, h.Malih, dll
    text = re.sub(r'(?<=\b[hH])\.(?=[A-Z])', '. ', text)

    # 7️⃣ Hapus simbol tidak diinginkan (tetap pertahankan tanda baca penting)
   
    text = re.sub(r'[^\w\s.,!?;:—\-❤️🩷🩵🟢🟡🟠🟣🟤💛💚💙💜🖤💖💘💝💞💟💌🎯🎉🎁🚀✨❤]', '', text)

    # 8️⃣ Rapikan spasi dan titik ganda
   
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\.{2,}', '.', text)

    # 9️⃣ Tambahkan spasi setelah tanda baca lain jika menempel huruf
    text = re.sub(r'([.,!?;:])(?=\w)', r'\1 ', text)

    
    # 6️⃣ Lowercase
    text = text.lower()

    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    
    return text


# In[13]:


# for i, comment in enumerate(comments, 1):
#    cleaned = clean_text_for_nlp(comment)
#    print(f"Original {i}: {comment}")
#    print(f"Cleaned {i}:  {cleaned}")
#    print("-" * 50)

#    df['cleaned_comment_text'] = cleaned



# In[ ]:


df['cleaned_comment_text'] = df['comment_text'].progress_apply(clean_text_for_nlp)


# In[ ]:


df = df[df['cleaned_comment_text'].str.strip() != '']


# In[ ]:


df = df[df['cleaned_comment_text'].str.split().str.len() > 1]


# In[ ]:


df = df[~df['cleaned_comment_text'].str.contains(r'\b\d{1,2}:\d{2}\b')]


# In[ ]:


df = df[~df['cleaned_comment_text'].str.fullmatch(r'\d+')]


# In[ ]:


df = df.reset_index(drop=True)


# In[ ]:


df.to_csv("cleaned_comments.csv", index=False)

