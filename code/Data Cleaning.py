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

class JudolTextCleaner:
    def __init__(self, domain_number_strategy='preserve', number_replacement_strategy='smart'):
        """
        Initialize JudolTextCleaner
        
        Parameters:
        domain_number_strategy (str): Strategi untuk angka di domain
            - 'remove': Hapus angka di akhir (pstoto99 -> pstoto)
            - 'preserve': Pertahankan angka sebagai token terpisah (pstoto99 -> pstoto 99) [RECOMMENDED]
            - 'separate_token': Gunakan token khusus (pstoto99 -> pstoto [DOMAIN_NUMBER])
        number_replacement_strategy (str): Strategi untuk angka di tengah kata
            - 'aggressive': Ganti semua angka dengan huruf (insan4d -> insanad)
            - 'smart': Ganti hanya angka yang membentuk kata judol, pertahankan lainnya (insan4d -> insan4d) [RECOMMENDED]
            - 'preserve': Pertahankan semua angka asli
        """       

        # Initialize Sastrawi untuk bahasa Indonesia
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
        stopword_factory = StopWordRemoverFactory()
        self.stopword_remover = stopword_factory.create_stop_word_remover()
        
        # Strategy configuration
        self.domain_number_strategy = domain_number_strategy
        self.number_replacement_strategy = number_replacement_strategy
        
        # Mapping untuk number replacement
        self.number_map = {
            '0': 'o', '1': 'i', '2': 'z', '3': 'e', '4': 'a',
            '5': 's', '6': 'g', '7': 't', '8': 'b', '9': 'g',
            '!': 'i', '@': 'a', '$': 's', '+': 't'
        }

        # Emoji mapping untuk angka dan huruf
        self.emoji_numbers = {
            '1️⃣': '1', '2️⃣': '2', '3️⃣': '3', '4️⃣': '4', '5️⃣': '5', 
            '6️⃣': '6', '7️⃣': '7', '8️⃣': '8', '9️⃣': '9', '0️⃣': '0',
            '➀': '1', '➁': '2', '➂': '3', '➃': '4', '➄': '5',
            '➅': '6', '➆': '7', '➇': '8', '➈': '9', '🄋': '0',
            '🥇': '1', '🥈': '2', '🥉': '3', '🏆': ' trophy ', '🎯': ' target ',
            '❶': '1', '❷': '2', '❸': '3', '❹': '4', '❺': '5',
            '❻': '6', '❼': '7', '❽': '8', '❾': '9', '❿': '10',
        }
        
        self.emoji_letters = {
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

        # Leet speak patterns khusus bahasa Indonesia
        self.leet_speak_indonesia = {
            'b9s4n': 'bosan', 'b0s4n': 'bosan', 'b05an': 'bosan',
            '7un9kad': 'rungkad', 'tun9kad': 'rungkad', 'rungk4d': 'rungkad',
            'runk4d': 'rungkad', 'j04n': 'join', 'j01n': 'join',
            'buru4n': 'buruan', 'b3s4r': 'besar', 'k3c11': 'kecill',
            'murah4n': 'murahan', 'g4c0r': 'gacor', 'g4cor': 'gacor',
            'm4nt4p': 'mantap', 'm4nt4b': 'mantap', 's3r1u': 'serius',
            's3r1ous': 'serius', 'wd': 'wd', 'dp': 'deposit', 'd3p0': 'depo',
            'm4nd4ng': 'mandang', 's4k1t': 'sakit', 'b4ng3t': 'banget',
            'k3r3n': 'keren', 'h4ncur': 'hancur', 'm4ntu1': 'mantul',
            'p4st1': 'pasti', 't0p': 'top', 'pr0': 'pro', 'n0ob': 'noob',
            'c0b4': 'coba',
        }
        
        # Extended character normalization mapping - DIPERLUAS dan DIPERBAIKI
        self.extended_char_map = {
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

        # Common judol domains untuk pattern recognition
        self.judol_domains = [
            'pstoto', 'toto', 'slot', 'poker', 'judi', 'bonus', 'arwana', 
            'pulau', 'win', 'casino', 'situs', 'bandar', 'sabung', 'taruhan',
            'insan', 'lazadatoto', 'paste4d', 'pandora4d', 'naga4d', 'hoki4d',
            'sendal4d', 'garudahoki', 'togel62', 'arwanatoto', 'pstoto99',
            'sgi88', 'sgi', 'sg188', 'sgi808', 'sgi888', 'sekali4d'  # TAMBAHKAN SEKALI4D
        ]

        # Brand names yang harus dipertahankan sebagai SATU KATA - DIPERBAIKI
        self.preserved_brands = {
            'insan4d': 'insan4d', 'pandora4d': 'pandora4d', 'naga4d': 'naga4d',
            'hoki4d': 'hoki4d', 'jaya4d': 'jaya4d', 'mega4d': 'mega4d',
            'super4d': 'super4d', 'lazadatoto': 'lazadatoto', 'lazada4d': 'lazada4d',
            'lazada88': 'lazada88', 'lazada77': 'lazada77', 'paste4d': 'paste4d',
            'pstoto99': 'pstoto99', 'pstoto88': 'pstoto88', 'pstoto77': 'pstoto77',
            'arwanatoto': 'arwanatoto', 'pulauwin': 'pulauwin', 'sendal4d': 'sendal4d',
            'garudahoki': 'garudahoki', 'togel62': 'togel62', 
            'sgi88': 'sgi88', 'sg188': 'sg188', 'sgi808': 'sgi808', 'sgi888': 'sgi888',
            'pstoto': 'pstoto', 'sekali4d': 'sekali4d'  # TAMBAHKAN SEKALI4D
        }

        # Common judol words untuk reconstruction - DIPERBAIKI
        self.judol_words_for_reconstruction = [
            'pulauwin', 'pulau', 'win', 'arwanatoto', 'arwana', 'toto',
            'lazadatoto', 'lazada', 'pstoto', 'pstoto99', 'pstoto88', 'pstoto77',
            'insan4d', 'pandora4d', 'paste4d', 'situs', 'slot', 'judi', 'togel', 
            'poker', 'bonus', 'deposit', 'withdraw', 'jackpot', 'freespin', 
            'casino', 'bandar', 'sabung', 'taruhan', 'rungkad', 'bosan', 'join', 
            'buruan', 'gacor', 'mantap', 'sendal4d', 'garudahoki', 'garuda', 
            'hoki', 'togel62', 'sgi88', 'sgi', 'sg188', 'sgi808', 'sgi888',
            'sekali4d'  # TAMBAHKAN SEKALI4D
        ]

        # Words yang mengandung angka tapi harus dipertahankan (brand names dengan angka) - DIPERBAIKI
        self.preserve_number_words = {
            'sendal4d', 'insan4d', 'pandora4d', 'naga4d', 'hoki4d', 'jaya4d',
            'mega4d', 'super4d', 'lazada4d', 'paste4d', 'pstoto99', 'pstoto88', 'pstoto77',
            'lazada88', 'garudahoki', 'togel62', 
            'sgi88', 'sg188', 'sgi808', 'sgi888', 'sekali4d'  # TAMBAHKAN SEKALI4D
        }

        # Common word combinations yang sering dipisah - DIPERBAIKI
        self.common_combinations = {
            'ga ruda ho ki': 'garudahoki',
            'ga ruda hoki': 'garudahoki',
            'garuda ho ki': 'garudahoki',
            'garuda hoki': 'garudahoki',
            'ga rudahoki': 'garudahoki',
            'pula uwin': 'pulauwin',
            'pulau win': 'pulauwin',
            'arwana toto': 'arwanatoto',
            'arwana to to': 'arwanatoto',
            'lazada toto': 'lazadatoto',
            'sendal 4d': 'sendal4d',
            'insan 4d': 'insan4d',
            'togel 62': 'togel62',
            'psto to': 'pstoto',
            'pstoto 99': 'pstoto99',
            'ps toto': 'pstoto',
            'sgi 88': 'sgi88',
            'sg 188': 'sg188',
            'sgi 808': 'sgi808',
            'sgi 888': 'sgi888',
            'di sgi88': 'di sgi88',
            'di pstoto99': 'di pstoto99',
            'sekali 4d': 'sekali4d',  # TAMBAHKAN SEKALI4D
        }

        # IMPORTANT WORDS yang TIDAK BOLEH dihapus oleh stopword removal
        self.important_words = {
            'saya', 'kamu', 'dia', 'kami', 'kita', 'mereka', 'ini', 'itu',
            'harapan', 'cuman', 'hanya', 'sekali', 'selalu', 'pernah', 'ingin',
            'mau', 'akan', 'bisa', 'dapat', 'boleh', 'harus', 'perlu', 'bisa',
            'membuat', 'menjadi', 'ubah', 'transformasi', 'dari', 'jadi',
            'pengantar', 'surat', 'manajer', 'direktur', 'bos', 'ketua',
            'manager', 'karyawan', 'pegawai', 'kerja', 'pekerjaan',
            'transaksi', 'deposit', 'withdraw', 'bonus', 'jackpot',
            'menang', 'kalah', 'untung', 'rugi', 'profit', 'hasil',
            'bertransformasi', 'berubah', 'hidup', 'nasib', 'kehidupan',
            'kaya', 'miskin', 'sukses', 'gagal', 'berhasil', 'pendapatan',
            'penghasilan', 'gaji', 'uang', 'duit', 'modal', 'investasi'
        }

        # Custom stopword list yang lebih selektif
        self.custom_stopwords = {
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

    # ===== IMPROVED STOPWORD REMOVAL =====
    def selective_stopword_removal(self, text):
        """Stopword removal yang selektif - hanya menghapus stopwords umum"""
        words = text.split()
        filtered_words = []
        
        for word in words:
            # Jangan hapus jika:
            # 1. Termasuk important words
            # 2. Adalah brand/judol word  
            # 3. Mengandung angka
            # 4. Panjang kata > 3 karakter
            # 5. Bukan stopword custom
            word_lower = word.lower()
            if (word_lower in self.important_words or
                any(brand in word_lower for brand in self.preserved_brands) or
                any(char.isdigit() for char in word) or
                len(word) > 3 or
                word_lower not in self.custom_stopwords):
                filtered_words.append(word)
        
        return ' '.join(filtered_words)

    # ===== IMPROVED WORD SEGMENTATION =====
    def improved_word_segmentation(self, text):
        """Segmentasi kata yang lebih baik untuk kasus seperti 'disgi88membuat'"""
        # Pattern untuk memisahkan kata yang menempel pada brand
        patterns = [
            # Kasus: prefix + brand + kata (disgi88membuat -> di sgi88 membuat)
            (r'(\b\w{1,2})(sgi88|sg188|sgi808|sgi888)(\w+)\b', r'\1 \2 \3'),
            (r'(\b\w{1,2})(pstoto|arwanatoto|garudahoki)(\w+)\b', r'\1 \2 \3'),
            
            # Kasus: kata + brand (membuatsgi88 -> membuat sgi88)
            (r'(\b\w+)(sgi88|sg188|sgi808|sgi888)(\w{1,2}\b)', r'\1 \2 \3'),
            (r'(\b\w+)(pstoto|arwanatoto|garudahoki)(\w{1,2}\b)', r'\1 \2 \3'),
            
            # Kasus: brand langsung gabung dengan kata
            (r'\b(sgi88|sg188|sgi808|sgi888)(\w{3,})\b', r'\1 \2'),
            (r'\b(\w{3,})(sgi88|sg188|sgi808|sgi888)\b', r'\1 \2'),
            
            # ✅ PERBAIKI: Pattern untuk Togel62, Sendal4d, Sekali4d - LEBIH SPESIFIK
            # Kasus: kata + togel62 (membuattogel62 -> membuat togel62)
            (r'\b(\w{3,})(togel62)(\w*)\b', r'\1 \2 \3'),
            # Kasus: togel62 + kata (togel62membuat -> togel62 membuat)  
            (r'\b(togel62)(\w{3,})\b', r'\1 \2'),
            # Kasus: prefix pendek + togel62 (ditogel62 -> di togel62)
            (r'\b(\w{1,2})(togel62)(\w*)\b', r'\1 \2 \3'),
            
            # Pattern yang sama untuk sendal4d dan sekali4d
            (r'\b(\w{3,})(sendal4d)(\w*)\b', r'\1 \2 \3'),
            (r'\b(sendal4d)(\w{3,})\b', r'\1 \2'),
            (r'\b(\w{1,2})(sendal4d)(\w*)\b', r'\1 \2 \3'),
            
            (r'\b(\w{3,})(sekali4d)(\w*)\b', r'\1 \2 \3'),
            (r'\b(sekali4d)(\w{3,})\b', r'\1 \2'),
            (r'\b(\w{1,2})(sekali4d)(\w*)\b', r'\1 \2 \3'),
            
            # Kasus umum: prefix + kata
            (r'\b(di)(\w{3,})\b', r'\1 \2'),  # dimembuat -> di membuat
            (r'\b(ke)(\w{3,})\b', r'\1 \2'),  # kemana -> ke mana
            (r'\b(se)(\w{3,})\b', r'\1 \2'),  # semahal -> se mahal
        ]
        
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text

    def fix_broken_alphanumeric(self, text):
        """
        Gabungkan huruf yang salah terpisah sebelum angka,
        misal 'se ru69' -> 'seru69', 'ju di777' -> 'judi777',
        tapi jangan gabungkan stopword seperti 'di', 'ke', 'yang', dll.
        """
        avoid_words = r'(di|ke|yang|aja|lagi|dan|itu|nya|bos|slot|jackpot|main|udah)'
        
        # Pastikan brand tetap utuh (tidak tergabung)
        for brand in getattr(self, "preserved_brands", []):
            text = re.sub(
                rf'\b([a-z]+)\s+({brand})\b',
                r'\1 \2',
                text,
                flags=re.IGNORECASE
            )
        
        # 🔧 1. Gabungkan tiga token sebelum angka (contoh: se ru 69 -> seru69)
        text = re.sub(
            r'\b([a-z]{1,3})\s+([a-z]{1,3})\s+([a-z]{1,3})(?=\d)',
            lambda m: m.group(1) + m.group(2) + m.group(3),
            text
        )
        
        # 🔧 2. Gabungkan dua token sebelum angka (contoh: se ru69 -> seru69)
        text = re.sub(
            r'\b([a-z]{1,4})\s+([a-z]{1,4})(?=\d)',
            lambda m: m.group(1) + m.group(2),
            text
        )
        
        # 🔧 3. Gabungkan angka yang terpisah (6 9 -> 69)
        text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
        
        return text



    def fix_spaced_letters_with_numbers(self, text):
        """
        Gabungkan huruf-huruf yang terpisah satu-satu diikuti angka.
        Contoh: 's u k u 8 8' -> 'suku88'
        """
        # Gabungkan huruf yang terpisah satu spasi
        text = re.sub(r'\b(?:([a-z])\s+){2,}([a-z])\b', 
                      lambda m: ''.join(m.group(0).split()), 
                      text)
        # Gabungkan angka yang terpisah satu spasi
        text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
        return text

    # ===== TEXT DECORATION CLEANING =====
    def remove_text_decorations(self, text):
        """Hapus dekorasi teks seperti |!¤*'~``~'*¤!| dan sejenisnya"""
        # Pattern untuk decorated text dengan berbagai simbol
        decoration_patterns = [
            r'[|!¤*\'~`¯,¸øº°∙▪■□▢▣▤▥▦▧▨▩▪▫▬▭▮▯▰▱▲△▴▵▶▷▸▹►▻▼▽▾▿◀◁◂◃◄◅◆◇◈◉◊○◌◍◎●◐◑◒◓◔◕◖◗◘◙◚◛◜◝◞◟◠◡◢◣◤◥◦◧◨◩◪◫◬◭◮◯◰◱◲◳◴◵◶◷◸◹◺◻◼◽◾◿]+',
        ]
        
        for pattern in decoration_patterns:
            text = re.sub(pattern, ' ', text)
        
        return text

    # ===== EMOJI HANDLING METHODS =====
    def replace_emoji_numbers(self, text):
        """Ganti emoji angka dengan angka biasa"""
        for emoji_num, normal_num in self.emoji_numbers.items():
            text = text.replace(emoji_num, normal_num)
        return text

    def replace_emoji_letters(self, text):
        """Ganti emoji huruf dengan huruf biasa"""
        for emoji_letter, normal_letter in self.emoji_letters.items():
            text = text.replace(emoji_letter, normal_letter)
        return text

    def handle_emoji_characters(self, text):
        """Handle semua jenis emoji karakter"""
        # Step 1: Replace emoji numbers
        text = self.replace_emoji_numbers(text)
        
        # Step 2: Replace emoji letters  
        text = self.replace_emoji_letters(text)
        
        # Step 3: Remove remaining emojis, tapi pertahankan makna
        text = emoji.demojize(text)
        text = re.sub(r':[a-z_]+:', ' ', text)  # Hapus kode emoji
        
        return text

    # ===== BASIC CLEANING METHODS =====
    def remove_emojis(self, text):
        """Hapus semua emoji dan simbol"""
        return emoji.replace_emoji(text, replace=' ')

    def fix_encoding(self, text):
        """Perbaiki encoding issues"""
        return ftfy.fix_text(text)

    def normalize_unicode(self, text):
        """Normalisasi karakter unicode"""
        return unicodedata.normalize('NFKD', text)

    def remove_special_chars(self, text):
        """Hapus karakter khusus tapi pertahankan huruf Indonesia"""
        # Pertahankan kata dengan angka (brand names)
        text = re.sub(r'[^\w\s\d]', ' ', text)
        return text

    def remove_urls(self, text):
        """Hapus URL dan domain"""
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'\S+\.(com|net|org|id|io)\S*', '', text)
        return text

    def remove_phone_numbers(self, text):
        """Hapus nomor telepon"""
        text = re.sub(r'[\+]?[0-9]{2,}[\s\-]?[0-9]{2,}[\s\-]?[0-9]{2,}[\s\-]?[0-9]{2,}', '', text)
        return text

    def clean_whitespace_characters(self, text):
        """Bersihkan karakter whitespace tidak terlihat"""
        text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
        text = re.sub(r'&nbsp;', ' ', text)
        return text

    # ===== IMPROVED CHARACTER NORMALIZATION =====
    def enhanced_character_normalization(self, text):
        """Normalisasi karakter extended dan khusus dengan lebih baik"""
        # Step 1: Handle emoji characters first
        text = self.handle_emoji_characters(text)
        
        # Step 2: Normalize Unicode (NFKD untuk memisahkan diacritics)
        text = unicodedata.normalize('NFKD', text)
        
        # Step 3: Replace extended characters
        for char, replacement in self.extended_char_map.items():
            text = text.replace(char, replacement)
        
        # Step 4: Remove diacritics (accents)
        text = ''.join(c for c in text if not unicodedata.combining(c))
        
        # Step 5: Use unidecode untuk karakter yang tersisa
        text = unidecode(text)
        
        # Step 6: Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def clean_brackets_and_special_chars(self, text):
        """Bersihkan brackets dan karakter khusus secara terpisah"""
        # Hapus semua brackets dan karakter khusus, ganti dengan spasi
        text = re.sub(r'[【】『』〖〗「」｢｣〔〕〈〉《»«〝〞＂‟〟：；，。、！？～‧・¢@®©™]', ' ', text)
        return text

    # ===== IMPROVED BRAND RECOGNITION =====
    def enhanced_brand_recognition(self, text):
        """Enhanced brand recognition dengan pattern matching yang lebih kuat"""
        brand_patterns = {
            # SGI88 variations
            r'\b[sS5][gG9][iI1]88\b': 'sgi88',
            r'\b[sS5][gG9][iI1]\s*88\b': 'sgi88', 
            r'\b[sS5][gG9][iI1]808\b': 'sgi808',
            r'\b[sS5][gG9][iI1]888\b': 'sgi888',
            r'\b[sS5][gG9]188\b': 'sg188',
            r'\b[sS5][gG9]\s*188\b': 'sg188',
            
            # PSTOTO99 variations
            r'\b[pP][sS5][tT7][oO0][tT7][oO0]99\b': 'pstoto99',
            r'\b[pP][sS5][tT7][oO0][tT7][oO0]\s*99\b': 'pstoto99',
            r'\bpstoto\s*99\b': 'pstoto99',
            
            # ✅ PERBAIKI: Togel62 variations - HAPUS pattern spacing di sini
            r'\b[tT7][oO0][gG9][eE3][lL1]62\b': 'togel62',
            r'\b[tT7][oO0][gG9][eE3][lL1]\s*62\b': 'togel62',
            
            # Sendal4d variations
            r'\b[sS5][eE3][nN][dD][aA4@][lL1]4[dD]\b': 'sendal4d',
            r'\b[sS5][eE3][nN][dD][aA4@][lL1]\s*4[dD]\b': 'sendal4d',
            
            # Sekali4d variations
            r'\b[sS5][eE3][kK][aA4@][lL1][iI1]4[dD]\b': 'sekali4d',
            r'\b[sS5][eE3][kK][aA4@][lL1][iI1]\s*4[dD]\b': 'sekali4d',
            
            # Pattern lainnya tetap sama...
        }
        
        for pattern, replacement in brand_patterns.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text

    # ===== BRAND PATTERN HANDLING =====
    def fix_specific_brand_patterns(self, text):
        """Perbaiki pattern brand khusus"""
        # Pattern untuk brackets dan special characters - HAPUS saja
        bracket_patterns = [
            r'【.*?】', r'『.*?』', r'〖.*?〗', r'「.*?」', r'｢.*?｣',
            r'〔.*?〕', r'〈.*?〉', r'《.*?》', r'«.*?»', r'@¢', r'®', r'©', r'™'
        ]
        
        for pattern in bracket_patterns:
            text = re.sub(pattern, ' ', text)
        
        # Pattern untuk brand names - NORMALIZE tapi pertahankan sebagai SATU KATA
        brand_patterns = {
            # SGI88 patterns - DIPERBAIKI
            r'sgi\s*88': 'sgi88',
            r'sg\s*188': 'sg188', 
            r'sgi\s*808': 'sgi808',
            r'sgi\s*888': 'sgi888',
            
            # PSTOTO99 patterns - DIPERBAIKI
            r'pstoto\s*99': 'pstoto99',
            r'ps\s*toto\s*99': 'pstoto99',
            
            # Togel62 patterns - TAMBAHKAN
            r'togel\s*62': 'togel62',
            r't0gel\s*62': 'togel62',
            
            # Sendal4d patterns - TAMBAHKAN
            r'sendal\s*4d': 'sendal4d',
            r'sendal\s*4\s*d': 'sendal4d',
            
            # Sekali4d patterns - TAMBAHKAN
            r'sekali\s*4d': 'sekali4d',
            r'sekali\s*4\s*d': 'sekali4d',
            
            # "cari di google" -> pisah menjadi 3 kata terpisah
            r'cari\s*di\s*google': 'cari di google',
            r'cari\s*di\s*g[o0][o0]gle': 'cari di google',
            r'çäri\s*di\s*göögle': 'cari di google',
            r'çari\s*di\s*google': 'cari di google',
            
            # "lazadatoto" -> pertahankan sebagai SATU KATA
            r'lazada\s*toto': 'lazadatoto',
            r'lazada\s*t[o0]t[o0]': 'lazadatoto',
            r'lazada\s*4d': 'lazada4d',
            
            # "garudahoki" -> pertahankan sebagai SATU KATA
            r'ga\s*ruda\s*ho\s*ki': 'garudahoki',
            r'ga\s*ruda\s*hoki': 'garudahoki',
            r'garuda\s*ho\s*ki': 'garudahoki',
            r'garuda\s*hoki': 'garudahoki',
        }
        
        for pattern, replacement in brand_patterns.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text

    def fix_brand_spacing(self, text):
        """Perbaiki spacing khusus untuk brand names - VERSI DIPERBAIKI"""
        # Pattern untuk memastikan brand names memiliki spasi yang tepat
        brand_spacing_patterns = {
            # ✅ PERBAIKI: Gunakan word boundaries dan pastikan spasi konsisten
            r'\b(\w{2,})(togel62|sendal4d|sekali4d|sgi88|sg188|sgi808|sgi888|pstoto99)\b': r'\1 \2',
            r'\b(togel62|sendal4d|sekali4d|sgi88|sg188|sgi808|sgi888|pstoto99)(\w{2,})\b': r'\1 \2',
            
            # Handle kasus khusus dengan karakter tunggal
            r'\b(\w{1})(togel62|sendal4d|sekali4d)\b': r'\1 \2',
            r'\b(togel62|sendal4d|sekali4d)(\w{1})\b': r'\1 \2',
        }
        
        for pattern, replacement in brand_spacing_patterns.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text

    def preserve_brand_names_in_text(self, text):
        """Pertahankan brand names sebagai satu kata dalam teks - VERSI DIPERBAIKI"""
        # ✅ PERBAIKI: Jangan tambahkan brand baru di sini, gunakan yang sudah ada di __init__
        
        # Normalize spacing terlebih dahulu
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Urutkan dari yang terpanjang ke terpendek untuk menghindari partial matching
        sorted_brands = sorted(self.preserved_brands.keys(), key=len, reverse=True)
        
        for brand in sorted_brands:
            preserved_form = self.preserved_brands[brand]
            
            # ✅ PERBAIKI: Gunakan pattern yang lebih spesifik
            # Pattern 1: Brand sebagai kata utuh dengan boundaries
            pattern1 = r'\b' + re.escape(brand) + r'\b'
            text = re.sub(pattern1, preserved_form, text, flags=re.IGNORECASE)
            
            # Pattern 2: Brand dengan spasi internal (sgi 88 -> sgi88)
            if any(char.isdigit() for char in brand):
                # Untuk brand dengan angka, buat pattern dengan optional spaces
                chars = list(brand)
                spaced_pattern = r'\s*'.join(re.escape(char) for char in chars)
                pattern2 = r'\b' + spaced_pattern + r'\b'
                text = re.sub(pattern2, preserved_form, text, flags=re.IGNORECASE)
        
        return text

    # ===== IMPROVED WORD COMBINATION FIXING =====
    def fix_common_combinations(self, text):
        """Perbaiki kombinasi kata yang sering dipisah"""
        # Normalize spacing terlebih dahulu
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Urutkan kombinasi dari yang terpanjang ke terpendek
        sorted_combinations = sorted(self.common_combinations.items(), 
                                   key=lambda x: len(x[0]), reverse=True)
        
        for combination, replacement in sorted_combinations:
            # Gunakan regex untuk matching yang lebih fleksibel
            pattern = r'\b' + re.escape(combination) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text

    # ===== IMPROVED NUMBER REPLACEMENT =====
    def smart_number_replacement(self, text):
        """Ganti angka dengan huruf hanya untuk kata-kata tertentu, pertahankan brand names"""
        # Step 1: Preserve brand names dengan angka terlebih dahulu
        text = self.preserve_brand_names_in_text(text)
        
        # Step 2: Untuk kata lainnya, gunakan replacement berdasarkan strategi
        if self.number_replacement_strategy == 'aggressive':
            # Ganti semua angka kecuali dalam brand names yang sudah dipreserve
            words = text.split()
            processed_words = []
            
            for word in words:
                # Jika word adalah brand name yang dipreserve, skip
                if word in self.preserved_brands.values():
                    processed_words.append(word)
                    continue
                
                # Jika word mengandung angka dan bukan brand name, lakukan replacement
                if any(char.isdigit() for char in word) and not word.isdigit():
                    processed_word = ''
                    for char in word:
                        if char in self.number_map:
                            processed_word += self.number_map[char]
                        else:
                            processed_word += char
                    processed_words.append(processed_word)
                else:
                    processed_words.append(word)
            
            text = ' '.join(processed_words)
            
        elif self.number_replacement_strategy == 'smart':
            # Hanya ganti angka pada kata-kata leet speak yang diketahui
            for leet_word, normal_word in self.leet_speak_indonesia.items():
                text = re.sub(r'\b' + re.escape(leet_word) + r'\b', normal_word, text, flags=re.IGNORECASE)
            
        # 'preserve' strategy tidak melakukan apa-apa terhadap angka
        
        return text

    def comprehensive_number_replacement(self, text):
        """Ganti angka dengan huruf yang sesuai - VERSI DIPERBAIKI"""
        # Step 1: Handle domain numbers berdasarkan strategi
        text = self.preserve_common_domains(text)
        
        # Step 2: Preserve brand names dengan angka
        text = self.preserve_brand_names_in_text(text)
        
        # Step 3: Decode leet speak Indonesia
        text = self.decode_leet_speak_indonesia(text)
        
        # Step 4: Smart number replacement berdasarkan strategi
        text = self.smart_number_replacement(text)
        
        return text

    # ===== WORD RECONSTRUCTION METHODS =====
    def advanced_word_reconstruction(self, text):
        """Rekonstruksi yang lebih advanced dengan pattern matching - DIPERBAIKI"""
        # Pattern untuk kata yang sering dipisah - DIPERBAIKI
        separation_patterns = {
            # SGI88 variations - DIPERBAIKI
            r'\b(s)\s*(g)\s*(i)\s*(8)\s*(8)\b': 'sgi88',
            r'\b(s\s*g\s*i\s*8\s*8)\b': 'sgi88',
            r'\b(sg)\s*(i88)\b': 'sgi88',
            r'\b(sgi)\s*(88)\b': 'sgi88',
            
            # PSTOTO99 variations - DIPERBAIKI
            r'\b(p)\s*(s)\s*(t)\s*(o)\s*(t)\s*(o)\s*(9)\s*(9)\b': 'pstoto99',
            r'\b(p\s*s\s*t\s*o\s*t\s*o\s*9\s*9)\b': 'pstoto99',
            r'\b(pstoto)\s*(99)\b': 'pstoto99',
            r'\b(ps)\s*(toto)\s*(99)\b': 'pstoto99',
            
            # Togel62 variations - TAMBAHKAN
            r'\b(t)\s*(o)\s*(g)\s*(e)\s*(l)\s*(6)\s*(2)\b': 'togel62',
            r'\b(t\s*o\s*g\s*e\s*l\s*6\s*2)\b': 'togel62',
            r'\b(togel)\s*(62)\b': 'togel62',
            
            # Sendal4d variations - TAMBAHKAN
            r'\b(s)\s*(e)\s*(n)\s*(d)\s*(a)\s*(l)\s*(4)\s*(d)\b': 'sendal4d',
            r'\b(s\s*e\s*n\s*d\s*a\s*l\s*4\s*d)\b': 'sendal4d',
            r'\b(sendal)\s*(4d)\b': 'sendal4d',
            
            # Sekali4d variations - TAMBAHKAN
            r'\b(s)\s*(e)\s*(k)\s*(a)\s*(l)\s*(i)\s*(4)\s*(d)\b': 'sekali4d',
            r'\b(s\s*e\s*k\s*a\s*l\s*i\s*4\s*d)\b': 'sekali4d',
            r'\b(sekali)\s*(4d)\b': 'sekali4d',
            
            # Pulauwin variations
            r'\b(p)\s*(u)\s*(l)\s*(a)\s*(u)\s*(w)\s*(i)\s*(n)\b': 'pulauwin',
            r'\b(p\s*u\s*l\s*a\s*u\s*w\s*i\s*n)\b': 'pulauwin',
            r'\b(pula)\s*(uwin)\b': 'pulauwin',
            r'\b(pulau)\s*(win)\b': 'pulauwin',
            
            # Arwanatoto variations
            r'\b(a)\s*(r)\s*(w)\s*(a)\s*(n)\s*(a)\s*(t)\s*(o)\s*(t)\s*(o)\b': 'arwanatoto',
            r'\b(arwana)\s*(toto)\b': 'arwanatoto',
            
            # Garudahoki variations
            r'\b(g)\s*(a)\s*(r)\s*(u)\s*(d)\s*(a)\s*(h)\s*(o)\s*(k)\s*(i)\b': 'garudahoki',
            r'\b(garuda)\s*(hoki)\b': 'garudahoki',
        }
        
        for pattern, replacement in separation_patterns.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text

    def detect_and_fix_word_separation(self, text):
        """Deteksi dan perbaiki pemisahan kata dengan algoritma yang lebih robust"""
        # Step 1: Fix common combinations first
        text = self.fix_common_combinations(text)
        
        # Step 2: Original algorithm
        words = text.split()
        if len(words) < 2:
            return text
        
        i = 0
        result_words = []
        
        while i < len(words):
            current_word = words[i]
            
            # Skip jika kata sudah panjang atau mengandung angka
            if len(current_word) > 3 or any(char.isdigit() for char in current_word):
                result_words.append(current_word)
                i += 1
                continue
            
            # Coba gabung dengan kata berikutnya
            if i + 1 < len(words):
                next_word = words[i + 1]
                combined = current_word + next_word
                combined_lower = combined.lower()
                
                # Cek apakah gabungan membentuk kata judol
                is_judol_word = any(
                    judol_word == combined_lower or 
                    judol_word.startswith(combined_lower) or
                    combined_lower.startswith(judol_word)
                    for judol_word in self.judol_words_for_reconstruction
                )
                
                if is_judol_word and len(combined) >= 4:
                    # Cari kata judol yang paling tepat
                    best_match = None
                    for judol_word in self.judol_words_for_reconstruction:
                        if judol_word.startswith(combined_lower) or combined_lower.startswith(judol_word):
                            best_match = judol_word
                            break
                    
                    if best_match:
                        result_words.append(best_match)
                        i += 2  # Skip kedua kata
                        continue
            
            result_words.append(current_word)
            i += 1
        
        return ' '.join(result_words)

    def reconstruct_separated_words(self, text):
        """Rekonstruksi kata yang sengaja dipisah seperti 'p u l a u w i n' atau 'pula uwin'"""
        words = text.split()
        
        if len(words) <= 1:
            return text
        
        reconstructed_words = []
        i = 0
        
        while i < len(words):
            current_word = words[i]
            
            # Coba gabung dengan kata berikutnya untuk membentuk kata judol
            combined = current_word
            j = i + 1
            found_combination = False
            
            while j <= len(words):
                # Cek kombinasi saat ini
                current_combination = ''.join(words[i:j])
                current_combination_lower = current_combination.lower()
                
                # Cek apakah kombinasi ini adalah kata judol atau bagian darinya
                is_judol_combination = any(
                    judol_word.startswith(current_combination_lower) or 
                    current_combination_lower in judol_word or
                    judol_word.startswith(current_combination_lower.replace(' ', ''))
                    for judol_word in self.judol_words_for_reconstruction
                )
                
                # Jika kombinasi membentuk kata judol yang lengkap
                exact_match = any(
                    judol_word == current_combination_lower.replace(' ', '')
                    for judol_word in self.judol_words_for_reconstruction
                )
                
                if exact_match:
                    # Found exact match! Gunakan kata judol yang benar
                    matched_word = next(
                        judol_word for judol_word in self.judol_words_for_reconstruction 
                        if judol_word == current_combination_lower.replace(' ', '')
                    )
                    reconstructed_words.append(matched_word)
                    i = j
                    found_combination = True
                    break
                elif is_judol_combination and j < len(words):
                    # Masih mungkin bisa digabung lebih lanjut
                    j += 1
                else:
                    # Tidak bisa digabung lebih lanjut
                    break
            
            if not found_combination:
                # Jika tidak ada kombinasi yang ditemukan, gunakan kata asli
                reconstructed_words.append(current_word)
                i += 1
        
        return ' '.join(reconstructed_words)

    # ===== LEET SPEAK DECODING =====
    def decode_leet_speak_indonesia(self, text):
        """Decode leet speak khusus bahasa Indonesia"""
        # Step 1: Replace known leet speak patterns
        for leet_word, normal_word in self.leet_speak_indonesia.items():
            text = re.sub(r'\b' + re.escape(leet_word) + r'\b', normal_word, text, flags=re.IGNORECASE)
        
        return text

    def smart_contextual_replacement(self, word):
        """Ganti angka dengan huruf berdasarkan konteks - HANYA untuk non-brand words"""
        # Jika word adalah brand name yang dipreserve, return asli
        if word.lower() in [brand.lower() for brand in self.preserved_brands.values()]:
            return word
        
        if not any(char.isdigit() for char in word) or word.isdigit():
            return word
        
        # Convert to lowercase untuk processing
        word_lower = word.lower()
        result = []
        i = 0
        
        while i < len(word_lower):
            char = word_lower[i]
            
            if char in self.number_map:
                # Special case untuk '7' (bisa 't' atau 'r')
                if char == '7':
                    # '7un9kad' -> 'rungkad' (7u -> ru)
                    if i + 1 < len(word_lower) and word_lower[i + 1] == 'u':
                        result.append('r')
                    else:
                        result.append('t')
                else:
                    result.append(self.number_map[char])
            else:
                result.append(char)
            i += 1
        
        return ''.join(result)

    def advanced_leet_decode(self, text):
        """Decode leet speak dengan algoritma yang lebih advanced"""
        words = text.split()
        decoded_words = []
        
        for word in words:
            # Skip jika sudah berupa brand yang di-preserve
            if word in self.preserved_brands.values():
                decoded_words.append(word)
                continue
            
            # Skip jika hanya angka
            if word.isdigit():
                decoded_words.append(word)
                continue
            
            # Coba decode dengan pattern yang diketahui dulu
            original_word = word
            word_lower = word.lower()
            
            # Decode known patterns
            for leet_pattern, normal_word in self.leet_speak_indonesia.items():
                if leet_pattern in word_lower:
                    word = word_lower.replace(leet_pattern, normal_word)
                    break
            
            # Jika masih ada angka dan BUKAN brand name, gunakan contextual replacement
            if any(char.isdigit() for char in word) and word_lower not in [brand.lower() for brand in self.preserved_brands.values()]:
                word = self.smart_contextual_replacement(word)
            
            decoded_words.append(word)
        
        return ' '.join(decoded_words)

    # ===== DOMAIN NUMBER HANDLING =====
    def handle_domain_numbers(self, text):
        """Handle angka di domain dengan strategi yang berbeda"""
        if self.domain_number_strategy == 'remove':
            for domain in self.judol_domains:
                pattern = r'\b(' + domain + r')(\d{2,3})\b'
                text = re.sub(pattern, r'\1', text, flags=re.IGNORECASE)
            
        elif self.domain_number_strategy == 'preserve':
            for domain in self.judol_domains:
                pattern = r'\b(' + domain + r')(\d{2,3})\b'
                text = re.sub(pattern, r'\1 \2', text, flags=re.IGNORECASE)
            
        elif self.domain_number_strategy == 'separate_token':
            for domain in self.judol_domains:
                pattern = r'\b(' + domain + r')(\d{2,3})\b'
                text = re.sub(pattern, r'\1 [DOMAIN_NUMBER]', text, flags=re.IGNORECASE)
            
        return text

    def preserve_common_domains(self, text):
        """Preserve common domain patterns yang mengandung angka"""
        return self.handle_domain_numbers(text)

    # ===== TEXT NORMALIZATION =====
    def normalize_case(self, text):
        """Normalisasi kapitalisasi"""
        return text.lower()

    def remove_stopwords_id(self, text):
        """Hapus stopwords bahasa Indonesia - GUNAKAN YANG SELEKTIF"""
        return self.selective_stopword_removal(text)

    def stem_text(self, text):
        """Stemming bahasa Indonesia"""
        return self.stemmer.stem(text)

    def remove_repeated_chars(self, text):
        """Kurangi karakter berulang berlebihan"""
        text = re.sub(r'([a-zA-Z])\1{2,}', r'\1\1', text)
        return text

    def handle_repeated_words(self, text):
        """Handle kata yang diulang-ulang"""
        text = re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', text)
        return text

    # ===== SPACING CLEANING =====
    def fix_advanced_spacing(self, text):
        """Perbaiki spacing"""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    # ===== SPECIAL CHARACTER CLEANING =====
    def clean_special_characters(self, text):
        """Bersihkan karakter khusus seperti @@ dan lainnya"""
        # Hapus karakter khusus seperti @@, **, dll
        text = re.sub(r'[@#\$%\^&\*\(\)_\+=\[\]\{\};:"\\|<>/~`]', ' ', text)
        # Hapus multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    # ===== IMPROVED CLEANING PIPELINE =====
    def clean_comprehensive(self, text, aggressive=True):
        """
        Pipeline cleaning komprehensif untuk teks judol - VERSI DIPERBAIKI
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # ✅ PERBAIKI: URUTAN YANG LEBIH OPTIMAL
        steps = [
            self.fix_encoding,
            self.remove_text_decorations,
            self.enhanced_character_normalization,
            self.clean_brackets_and_special_chars,
            self.handle_emoji_characters,
            self.clean_whitespace_characters,
            
            # ✅ FASE 1: Brand Recognition & Segmentation
            self.enhanced_brand_recognition,      # Kenali brand patterns
            self.improved_word_segmentation,      # Pisahkan kata yang menempel
            self.fix_brand_spacing,               # Pastikan spasi konsisten
            
            # ✅ FASE 2: Word Reconstruction  
            self.fix_common_combinations,         # Gabungkan kombinasi umum
            self.advanced_word_reconstruction,    # Rekonstruksi kata terpisah
            self.detect_and_fix_word_separation,  # Deteksi pemisahan kata
            self.reconstruct_separated_words,     # Rekonstruksi kata terpisah
            
            # ✅ FASE 3: Brand Preservation & Cleaning
            self.preserve_brand_names_in_text,    # ✅ DIPINDAH: Sekarang setelah reconstruction
            self.fix_specific_brand_patterns,     # Perbaiki pattern brand khusus
            
            # ✅ FASE 4: General Cleaning
            self.remove_urls,
            self.remove_phone_numbers,
            self.comprehensive_number_replacement,
            self.remove_special_chars,
            self.remove_repeated_chars,
            self.handle_repeated_words,
            self.normalize_case,
            self.fix_advanced_spacing,
            self.selective_stopword_removal,   # tambahkan di sini
            self.fix_spaced_letters_with_numbers, 
            self.fix_broken_alphanumeric,      # baru jalankan di sini
            self.stem_text,     
        ]
        
        # Add aggressive cleaning steps if enabled
        if aggressive:
            aggressive_steps = [
                self.selective_stopword_removal,
                self.stem_text,
                self.fix_advanced_spacing,
            ]
            steps.extend(aggressive_steps)
        
        # Execute cleaning pipeline
        cleaned_text = text
        print(f"Original: {text}")
        for step in steps:
            try:
                previous_text = cleaned_text
                cleaned_text = step(cleaned_text)
                
                # Debug: Cetak perubahan jika ada
                if previous_text != cleaned_text:
                    print(f"After {step.__name__}: {cleaned_text}")
                   
                    
                if not cleaned_text.strip():
                    return ""
            except Exception as e:
                print(f"Error in {step.__name__}: {e}")
                continue
                
        cleaned_text = self.preserve_brand_names_in_text(cleaned_text)

        print("-" * 100)
        return cleaned_text

    # ===== BATCH PROCESSING =====
    def clean_dataset(self, df, text_column='text', new_column='cleaned_text', aggressive=True):
        """
        Clean entire dataset
        
        Parameters:
        df (DataFrame): DataFrame pandas
        text_column (str): Nama kolom teks
        new_column (str): Nama kolom hasil cleaning
        aggressive (bool): Mode aggressive cleaning
        """
        tqdm.pandas(desc="Cleaning texts")
        df[new_column] = df[text_column].progress_apply(
            lambda x: self.clean_comprehensive(x, aggressive=aggressive)
        )
        
        return df

    # ===== ANALYSIS METHODS =====
    def analyze_cleaning_result(self, original_text, cleaned_text):
        """Analisis hasil cleaning"""
        analysis = {
            'original_length': len(original_text),
            'cleaned_length': len(cleaned_text),
            'reduction_ratio': round((len(original_text) - len(cleaned_text)) / len(original_text) * 100, 2) if original_text else 0,
            'original_words': len(original_text.split()),
            'cleaned_words': len(cleaned_text.split()),
            'contains_judol_keywords': self.contains_judol_keywords(cleaned_text)
        }
        return analysis

    def contains_judol_keywords(self, text):
        """Cek apakah teks mengandung kata kunci judol"""
        judol_keywords = [
            'situs', 'slot', 'judi', 'togel', 'poker', 'casino', 
            'taruhan', 'betting', 'deposit', 'withdraw', 'bonus',
            'jackpot', 'freespin', 'bandar', 'sabung', 'gambling',
            'pstoto', 'toto', 'arwana', 'pulauwin', 'lazadatoto',
            'insan4d', 'paste4d', 'pandora4d', 'bosan', 'rungkad',
            'sendal4d', 'garudahoki', 'togel62', 'arwanatoto', 'pstoto99',
            'sgi88', 'sg188', 'sgi808', 'sgi888', 'sekali4d'  # TAMBAHKAN SEKALI4D
        ]
        
        text_lower = text.lower()
        found_keywords = [keyword for keyword in judol_keywords if keyword in text_lower]
        return len(found_keywords) > 0, found_keywords


df = pd.read_csv('comments_from_scraping.csv')

cleaner = JudolTextCleaner()
 
for i, text in enumerate(df["comment_text"], 1):
    cleaned = cleaner.clean_comprehensive(text)
    
    df.loc[i-1, "comment_text"] = cleaned


empty_comments = df[df['comment_text'].str.strip() == '']
print(f"Jumlah komentar yang kosong: {len(empty_comments)}")
print(f"Persentase: {(len(empty_comments) / len(df)) * 100:.2f}%")



df = df[df['comment_text'].str.strip().astype(bool)].reset_index(drop=True)

output_filename = 'cleaned_comments.csv'
columns_to_save = [
    col for col in [
        'Unnamed: 0', 'comment_id', 'video_id', 'author', 
        'comment_text', 'published_at', 'like_count'
    ] if col in df.columns
]



df[columns_to_save].to_csv(output_filename, index=False, encoding='utf-8')
print(f"\n✅ Data ultimate disimpan ke: {output_filename}")
print(f"📊 Total baris: {len(df)}")

