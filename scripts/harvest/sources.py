"""Where the harvest looks. Three families, each with its own license evidence:

* ``YT_QUERIES`` — YouTube searches restricted to Creative-Commons uploads by the search filter
  (``sp=EgIwAQ==`` = "Features: Creative Commons"); every hit is still verified at fetch time from
  the info-json ``license`` field, which must read "Creative Commons Attribution license (reuse
  allowed)" (YouTube's only CC option is CC-BY 3.0). Queries carry the language they were issued in.
* ``USGOV_CHANNELS`` — official channels of U.S. federal agencies. Their own productions are works of
  the U.S. Government (17 U.S.C. § 105, no copyright). The info-json has no license field for these
  (YouTube shows "Standard YouTube License"), so the evidence recorded is the channel allowlist entry
  + the basis text + the channel id match verified at fetch time. NOT included: NIH VideoCast
  (visiting lecturers keep their copyright), party/caucus channels (not agencies), C-SPAN (private).
  VOA caveat: VOA's own productions are public domain; third-party newswire footage inside a VOA
  package is not — the talking-head prescreen keeps studio interviews/reports, and the caveat is
  written into every VOA clip's license_evidence.
* ``COMMONS_CATEGORIES`` / ``ARCHIVE_QUERIES`` — verified per file by ``scripts/fetch_sources.py``
  (Commons extmetadata, archive.org licenseurl); ND/NC/SA/missing are refused there.
"""
from __future__ import annotations

# (query, language) — the language is only a hint for guess_language when the title's script is ambiguous.
YT_QUERIES = [
    # English
    ("vlog", "en"), ("daily vlog", "en"), ("talking to camera", "en"), ("my story", "en"), ("storytime", "en"),
    ("interview", "en"), ("podcast episode", "en"), ("lecture", "en"), ("q&a", "en"), ("life update", "en"),
    ("day in my life", "en"), ("advice", "en"), ("motivational talk", "en"), ("sermon", "en"), ("testimony", "en"),
    ("book review", "en"), ("channel update", "en"), ("rant", "en"), ("opinion", "en"), ("commentary", "en"),
    ("teacher explains", "en"), ("doctor explains", "en"), ("lawyer explains", "en"), ("professor lecture", "en"),
    ("language lesson", "en"), ("english lesson", "en"), ("grwm chat", "en"), ("mental health talk", "en"),
    ("oral history interview", "en"), ("artist talk", "en"), ("author interview", "en"), ("startup pitch", "en"),
    ("conference talk", "en"), ("tutorial explained", "en"), ("news commentary", "en"), ("speech", "en"),
    ("reaction talking", "en"), ("student vlog", "en"), ("travel vlog talking", "en"), ("faith talk", "en"),
    # Spanish
    ("vlog diario", "es"), ("entrevista", "es"), ("hablando a la camara", "es"), ("mi historia", "es"),
    ("charla", "es"), ("podcast", "es"), ("clase", "es"), ("consejos", "es"), ("opinion", "es"), ("testimonio", "es"),
    # Portuguese
    ("vlog", "pt"), ("entrevista", "pt"), ("desabafo", "pt"), ("minha historia", "pt"), ("aula", "pt"), ("conversa", "pt"),
    # French
    ("vlog", "fr"), ("interview", "fr"), ("je vous parle", "fr"), ("mon histoire", "fr"), ("temoignage", "fr"),
    ("cours", "fr"), ("conseils", "fr"), ("podcast", "fr"),
    # German
    ("vlog", "de"), ("interview", "de"), ("meine geschichte", "de"), ("ich erzaehle", "de"), ("vortrag", "de"), ("podcast", "de"),
    # Italian
    ("vlog", "it"), ("intervista", "it"), ("vi racconto", "it"), ("lezione", "it"), ("chiacchiere", "it"),
    # Russian / Ukrainian
    ("влог", "ru"), ("интервью", "ru"), ("моя история", "ru"), ("разговор", "ru"), ("лекция", "ru"), ("советы", "ru"),
    ("влог", "uk"), ("інтерв'ю", "uk"), ("моя історія", "uk"),
    # Arabic / Persian / Urdu
    ("فلوق", "ar"), ("مقابلة", "ar"), ("قصتي", "ar"), ("نصائح", "ar"), ("محاضرة", "ar"),
    ("ولاگ", "fa"), ("مصاحبه", "fa"), ("داستان من", "fa"),
    ("ولاگ", "ur"), ("انٹرویو", "ur"), ("میری کہانی", "ur"),
    # Hindi / Bengali / Tamil
    ("व्लॉग", "hi"), ("इंटरव्यू", "hi"), ("मेरी कहानी", "hi"), ("बातचीत", "hi"), ("मोटिवेशन", "hi"),
    ("ভ্লগ", "bn"), ("সাক্ষাৎকার", "bn"), ("আমার গল্প", "bn"),
    ("வ்லாக்", "ta"), ("நேர்காணல்", "ta"),
    # Indonesian / Malay / Filipino / Vietnamese / Thai
    ("vlog", "id"), ("wawancara", "id"), ("cerita saya", "id"), ("curhat", "id"), ("ceramah", "id"),
    ("vlog", "tl"), ("kwento ko", "tl"), ("panayam", "tl"),
    ("vlog", "vi"), ("phỏng vấn", "vi"), ("tâm sự", "vi"), ("chia sẻ", "vi"),
    ("vlog", "th"), ("สัมภาษณ์", "th"), ("เล่าเรื่อง", "th"),
    # Turkish / Polish / Dutch / Greek / Swedish
    ("vlog", "tr"), ("röportaj", "tr"), ("hikayem", "tr"), ("sohbet", "tr"),
    ("vlog", "pl"), ("wywiad", "pl"), ("moja historia", "pl"),
    ("vlog", "nl"), ("interview", "nl"), ("mijn verhaal", "nl"),
    ("vlog", "el"), ("συνέντευξη", "el"),
    ("vlogg", "sv"), ("intervju", "sv"),
    # Japanese / Korean / Chinese
    ("vlog", "ja"), ("インタビュー", "ja"), ("雑談", "ja"), ("自己紹介", "ja"), ("講義", "ja"),
    ("브이로그", "ko"), ("인터뷰", "ko"), ("수다", "ko"), ("이야기", "ko"), ("강의", "ko"),
    ("vlog", "zh"), ("访谈", "zh"), ("聊天", "zh"), ("分享", "zh"), ("讲座", "zh"),
    # Swahili / Amharic / Hausa
    ("mahojiano", "sw"), ("hadithi yangu", "sw"), ("vlog", "sw"),
    ("ቃለ መጠይቅ", "am"), ("hira", "ha"),
]

# YouTube search "sp" params: features=Creative Commons, combined with three sort orders so the same
# query yields different pages over successive crawls.
YT_SP = {"relevance": "EgIwAQ%253D%253D", "date": "CAISAjAB", "views": "CAMSAjAB"}

USGOV_BASIS = ("Work of the U.S. Government: produced by federal employees in the course of their duties, "
               "not subject to copyright (17 U.S.C. § 105). Evidence: official agency channel (allowlist "
               "scripts/harvest/sources.py) + channel id verified on the fetched info-json.")
VOA_CAVEAT = ("VOA-produced content is public domain; third-party newswire material inside a VOA package is "
              "not. Only single-face talking-head segments pass the harvest prescreen.")

# (channel URL, agency, note or "")
USGOV_CHANNELS = [
    ("https://www.youtube.com/@WhiteHouse/videos", "The White House", ""),
    ("https://www.youtube.com/@statedept/videos", "U.S. Department of State", ""),
    ("https://www.youtube.com/@DeptofDefense/videos", "U.S. Department of Defense", ""),
    ("https://www.youtube.com/@NASA/videos", "NASA", ""),
    ("https://www.youtube.com/@USArmy/videos", "U.S. Army", ""),
    ("https://www.youtube.com/@USNavy/videos", "U.S. Navy", ""),
    ("https://www.youtube.com/@usairforce/videos", "U.S. Air Force", ""),
    ("https://www.youtube.com/@CDC/videos", "Centers for Disease Control and Prevention", ""),
    ("https://www.youtube.com/@USDA/videos", "U.S. Department of Agriculture", ""),
    ("https://www.youtube.com/@NOAA/videos", "NOAA", ""),
    ("https://www.youtube.com/@FEMA/videos", "FEMA", ""),
    ("https://www.youtube.com/@usgs/videos", "U.S. Geological Survey", ""),
    ("https://www.youtube.com/@NIST/videos", "NIST", ""),
    ("https://www.youtube.com/@VOANews/videos", "Voice of America", VOA_CAVEAT),
    ("https://www.youtube.com/@VOAAfrica/videos", "Voice of America (Africa)", VOA_CAVEAT),
    ("https://www.youtube.com/@VOAKhmer/videos", "Voice of America (Khmer)", VOA_CAVEAT),
    ("https://www.youtube.com/@VOATiengViet/videos", "Voice of America (Vietnamese)", VOA_CAVEAT),
    ("https://www.youtube.com/@VOABurmese/videos", "Voice of America (Burmese)", VOA_CAVEAT),
    ("https://www.youtube.com/@VOAAmharic/videos", "Voice of America (Amharic)", VOA_CAVEAT),
    ("https://www.youtube.com/@VOAIndonesia/videos", "Voice of America (Indonesian)", VOA_CAVEAT),
    ("https://www.youtube.com/@VOAUrdu/videos", "Voice of America (Urdu)", VOA_CAVEAT),
    ("https://www.youtube.com/@VOABangla/videos", "Voice of America (Bangla)", VOA_CAVEAT),
    ("https://www.youtube.com/@VOASwahili/videos", "Voice of America (Swahili)", VOA_CAVEAT),
    ("https://www.youtube.com/@VOAPersian/videos", "Voice of America (Persian)", VOA_CAVEAT),
    ("https://www.youtube.com/@VOAHausa/videos", "Voice of America (Hausa)", VOA_CAVEAT),
    ("https://www.youtube.com/@VOAKorea/videos", "Voice of America (Korean)", VOA_CAVEAT),
    ("https://www.youtube.com/@VOAChinese/videos", "Voice of America (Mandarin)", VOA_CAVEAT),
    ("https://www.youtube.com/@VOARussian/videos", "Voice of America (Russian)", VOA_CAVEAT),
    ("https://www.youtube.com/@VOASpanish/videos", "Voice of America (Spanish)", VOA_CAVEAT),
    ("https://www.youtube.com/@VOATurkce/videos", "Voice of America (Turkish)", VOA_CAVEAT),
]

# Verified per file by fetch_sources.classify_license on Commons extmetadata; unknown category names
# simply list nothing.
COMMONS_CATEGORIES = [
    {"cat": "Barack Obama weekly video addresses", "subcats": 1},
    {"cat": "Weekly address of the President of the United States", "subcats": 1},
    {"cat": "Interview videos", "subcats": 1},
    {"cat": "Video interviews", "subcats": 1},
    {"cat": "Scientist interview videos", "subcats": 1},
    {"cat": "Videos of Wikipedians", "subcats": 1},
    {"cat": "WikiDonne video interviews"},
    {"cat": "Vlog videos", "subcats": 1},
    {"cat": "Video blogs", "subcats": 1},
    {"cat": "Voice of America videos", "subcats": 1},
    {"cat": "Videos of lectures", "subcats": 1},
    {"cat": "Videos of speeches", "subcats": 1},
    {"cat": "Videos of people talking", "subcats": 1},
    {"cat": "TEDx videos", "subcats": 1},
    {"cat": "Wikimania videos", "subcats": 1},
    {"cat": "Videos of oral history", "subcats": 1},
    {"cat": "Video testimonies", "subcats": 1},
]

ARCHIVE_QUERIES = [
    'mediatype:movies AND (licenseurl:*publicdomain* OR licenseurl:*licenses/by/*) AND (title:interview) AND NOT collection:*television*',
    'mediatype:movies AND (licenseurl:*publicdomain* OR licenseurl:*licenses/by/*) AND (title:"oral history") AND NOT collection:*television*',
    'mediatype:movies AND (licenseurl:*publicdomain* OR licenseurl:*licenses/by/*) AND (title:lecture) AND NOT collection:*television*',
    'mediatype:movies AND (licenseurl:*publicdomain* OR licenseurl:*licenses/by/*) AND (title:vlog OR title:"video blog")',
    'mediatype:movies AND (licenseurl:*licenses/by/*) AND (subject:interview OR subject:talk) AND NOT collection:*television*',
]

# Titles that are almost never a single talking head; skipping them saves bandwidth, the prescreen
# would reject them anyway.
TITLE_BLOCK = (r"gameplay|minecraft|fortnite|roblox|walkthrough|playthrough|let'?s play|asmr|music video|"
               r"official video|lyrics|karaoke|remix|beat|instrumental|trailer|timelapse|time-lapse|drone|aerial|"
               r"highlights|compilation|montage|b-?roll|slideshow|screen recording|screencast|no commentary|"
               r"animation|animated|cartoon|unboxing|live stream|livestream|24/7|meditation|white noise|rain sounds|"
               r"\bmix\b|\bdj\b|concert|cooking|recipe|workout|yoga|dance|choreography|footage|nature|scenery|"
               r"\bloop\b|relaxing|sleep|study with me|pomodoro|coding session|speedrun|match|full game|"
               r"parade|ceremony|launch|liftoff|rover|telescope|satellite")
