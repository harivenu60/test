import streamlit as st
import pandas as pd
import requests, re, difflib, time
from datetime import datetime, timedelta
from transformers import pipeline
import nltk
import os

# -------------------------------
# CONFIGURATION / METADATA
# -------------------------------
st.set_page_config(page_title="Adverse News & Sanctions Search (v2)", page_icon="🛡️", layout="wide")

st.title("🛡️ Adverse News + Sanctions Search — (FinBERT default)")

# -------------------------------
# MODEL SELECTION (FinBERT default)
# -------------------------------
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox(
    "Select Model:",
    ["ProsusAI/finbert", "cardiffnlp/twitter-roberta-base-sentiment-latest"],
    index=0,
    help="FinBERT is default (financial/regulatory domain). You can switch to RoBERTa if you want."
)

@st.cache_resource
def load_sentiment_model(name):
    try:
        nltk.data.find('tokenizers/punkt')
    except Exception:
        nltk.download('punkt', quiet=True)
    # load pipeline (this will download model the first time)
    return pipeline("sentiment-analysis", model=name, tokenizer=name)

sentiment_model = load_sentiment_model(model_choice)
st.sidebar.success(f"✅ Sentiment model loaded: {model_choice}")

# -------------------------------
# API KEYS & Endpoints (kept as provided)
# -------------------------------
NEWSDATA_API_KEY = "pub_6f6a5665efae4a90823bc0195a8343f5"
NEWSAPI_API_KEY = "5b9afc75540c42a187ac83c8b61a165b"
GNEWS_API_KEY = "ee87c2b5d536c77dcbac6474f06497e8"

NEWSDATA_ENDPOINT = "https://newsdata.io/api/1/news"
NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"
GNEWS_ENDPOINT = "https://gnews.io/api/v4/search"

OFAC_URL = "https://sanctionslistservice.ofac.treas.gov/api/Publication"
UK_URL = "https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/1159796/UK_Sanctions_List.json"
OPENSANCTIONS_URL = "https://api.opensanctions.org/datasets/default/entities/"

# -------------------------------
# Utility helpers
# -------------------------------
def normalize_name(n):
    if not n:
        return ""
    n = n.lower()
    # remove common legal suffixes and noisy tokens
    remove_tokens = ['public joint stock company', 'pjsc', 'plc', 'llc', 'ltd', 'limited', 
                     'inc', 'corp', 'corporation', 'co.', 'company', 'sa', 'gmbh', 'pjsc', ',']
    for t in remove_tokens:
        n = n.replace(t, ' ')
    n = re.sub(r'\s+', ' ', n).strip()
    return n

def fuzzy_match(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

# -------------------------------
# Sanctions fetching (improved)
# -------------------------------
def fetch_ofac_list():
    try:
        resp = requests.get(OFAC_URL, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        names = []
        for entry in data.get("SDNList", {}).get("SDNEntries", []):
            # include lastName/firstName and entity/aka names
            if entry.get("lastName"):
                names.append(entry.get("lastName"))
            if entry.get("firstName"):
                names.append(entry.get("firstName"))
            if entry.get("sdnType"):
                # sometimes entity names are in 'lastName' or 'firstName' fields; also check alternate names
                pass
            # check aka list
            for aka in entry.get("akaList", {}).get("aka", []):
                if aka.get("akaName"):
                    names.append(aka.get("akaName"))
            # check other name fields
            for k in ['title', 'address', 'remarks']:
                if entry.get(k):
                    # try extract capitalized sequences as possible names - skip heavy parsing
                    pass
        # return unique normalized
        return list({n for n in names if n})
    except Exception as e:
        st.error(f"Error fetching OFAC list: {e}")
        return []

def fetch_uk_list():
    try:
        resp = requests.get(UK_URL, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        names = []
        for item in data:
            nm = item.get("Name") or item.get("name")
            if nm:
                names.append(nm)
        return list({n for n in names if n})
    except Exception as e:
        st.error(f"Error fetching UK sanctions list: {e}")
        return []

def fetch_opensanctions_all(limit=1000, max_items=5000):
    names = []
    try:
        offset = 0
        while True:
            url = OPENSANCTIONS_URL + f"?limit={limit}&offset={offset}"
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                break
            data = resp.json()
            results = data.get("results", [])
            if not results:
                break
            for r in results:
                nm = r.get("properties", {}).get("name")
                if nm:
                    names.append(nm)
            offset += limit
            if offset >= max_items:
                break
            time.sleep(0.1)
        return list({n for n in names if n})
    except Exception as e:
        st.error(f"Error fetching OpenSanctions: {e}")
        return []

# Small fallback list of high-risk entities (can be expanded)
FALLBACK_SANCTIONS = [
    "lukoil", "rosneft", "gazprom", "sberbank", "vtb", "alrosa", "gazprombank", "vtb bank"
]

# -------------------------------
# Expanded negative keywords (financial, sanctions, ESG, governance)
# -------------------------------
DEFAULT_NEGATIVE_KEYWORDS = [
    # Financial & AML
    "fraud", "laundering", "ponzi", "money laundering", "tax evasion", "embezzlement",
    "forgery", "kickback", "shell company", "suspicious transaction", "false accounting",
    "terror financing", "insider trading", "asset freeze", "illegal transfer", "bribery", "corruption",
    "diversion of funds", "misappropriation",

    # Legal & Enforcement
    "charges", "convicted", "indicted", "lawsuit", "criminal case", "penalty", "fine", "prosecuted",
    "guilty plea", "investigation", "probe", "raid", "arrested", "suspended", "debarred",
    "regulatory action", "compliance breach", "sanctioned",

    # Sanctions & International Risk
    "sanction", "ofac", "eu sanction", "uk sanction", "fatf", "blocked entity", "blacklisted",
    "restricted", "export ban", "embargo", "asset freeze", "designated entity", "designated vessel",
    "sdn", "sdn list", "blocked", "iran", "syria", "cuba", "north korea", "crimea", "donetsk", "luhansk",

    # ESG / Environmental & Governance
    "environmental violation", "oil spill", "pollution", "governance failure", "labor abuse",
    "human rights abuse", "workplace accident", "unethical practice", "illegal dumping",
    "bribery charges", "corruption probe", "compliance failure", "reputational risk"
]

# -------------------------------
# Highlight helper
# -------------------------------
def color_highlight_terms(text, name, keywords):
    if not text:
        return ""
    if name:
        text = re.sub(fr"(?i)\b({re.escape(name)})\b",
                      r"<span style='background-color: #fff176; font-weight:bold;'>\\1</span>", text)
    for kw in sorted(set(keywords), key=lambda s: -len(s)):
        text = re.sub(fr"(?i)\b({re.escape(kw)})\b",
                      r"<span style='background-color: #ef9a9a; font-weight:bold;'>\\1</span>", text)
    return text

# -------------------------------
# Main search & analysis function (uses transformer model)
# -------------------------------
def search_all(name, keywords, from_date, to_date, high_severity=False):
    # combine and dedupe keywords
    all_keywords = list({*(keywords + DEFAULT_NEGATIVE_KEYWORDS)})
    # build broad query (name + negative keywords)
    query = " ".join(([name] if name else []) + all_keywords)

    # fetch articles from news APIs
    articles = []
    articles += fetch_from_newsdata(query, from_date, to_date)
    articles += fetch_from_newsapi(query, from_date, to_date)
    articles += fetch_from_gnews(query, from_date, to_date)

    # fetch sanctions lists (once)
    sanctions_names = []
    sanctions_names += fetch_ofac_list()
    sanctions_names += fetch_uk_list()
    sanctions_names += fetch_opensanctions_all(limit=1000, max_items=4000)
    sanctions_names = list({n for n in sanctions_names if n})
    # add fallback
    sanctions_names += FALLBACK_SANCTIONS
    sanctions_names = list({n for n in sanctions_names if n})

    # debug info
    st.write(f"🔍 Loaded {len(sanctions_names)} sanctions entries (incl. fallback).")

    results = []
    # check sanctions matching for the name explicitly
    if name:
        name_norm = normalize_name(name)
        matches = []
        for s in sanctions_names:
            s_norm = normalize_name(s)
            score = fuzzy_match(name_norm, s_norm)
            if score >= 0.6:
                matches.append({"sanctioned_name": s, "similarity": round(score, 2)})
        # show sanctions matches first
        if matches:
            matches = sorted(matches, key=lambda x: -x["similarity"])
            for m in matches:
                results.append({
                    "title": f"Sanctions match: {m['sanctioned_name']}",
                    "source": "SanctionsLists",
                    "date": "",
                    "snippet_html": f"Possible sanctions match: <b>{m['sanctioned_name']}</b> (similarity: {m['similarity']})",
                    "url": "",
                    "score": m['similarity'],
                    "severity": "High",
                    "model_label": "SANCTIONS_MATCH"
                })

    # process news articles through transformer sentiment classifier
    for art in articles:
        if not isinstance(art, dict):
            continue
        text = f"{art.get('title','')} {art.get('desc','')}"
        if not text.strip():
            continue
        # run transformer sentiment (truncate)
        try:
            res = sentiment_model(text[:1024])[0]
            label = res.get('label','') or ''
            score = float(res.get('score',0.0))
        except Exception as e:
            label = "ERROR"
            score = 0.0
        # determine negativity
        lab_lower = label.lower()
        is_negative = False
        if 'neg' in lab_lower or 'negative' in lab_lower:
            is_negative = True
        # cardiffnlp mapping (LABEL_0 -> negative)
        if model_choice and 'twitter-roberta' in model_choice and label.upper().startswith('LABEL_'):
            if label.upper() == 'LABEL_0':
                is_negative = True
        # only keep negative articles
        if not is_negative:
            continue
        # severity mapping by score
        if score >= 0.85:
            severity = "High"
        elif score >= 0.6:
            severity = "Medium"
        else:
            severity = "Low"
        if high_severity and severity != "High":
            continue
        snippet_html = color_highlight_terms(text, name, all_keywords)
        results.append({
            "title": art.get("title",""),
            "source": art.get("source",""),
            "date": art.get("date",""),
            "snippet_html": snippet_html,
            "url": art.get("url",""),
            "score": score,
            "severity": severity,
            "model_label": label
        })
    return results

# -------------------------------
# News API helpers (kept similar to prior)
# -------------------------------
def fetch_from_newsdata(query, from_date, to_date):
    try:
        params = {"apikey": NEWSDATA_API_KEY, "q": query, "from_date": from_date, "to_date": to_date, "language": "en"}
        r = requests.get(NEWSDATA_ENDPOINT, params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        articles = []
        for a in j.get("results", []):
            articles.append({"title": a.get("title",""), "desc": a.get("description",""), "date": a.get("pubDate",""), "url": a.get("link",""), "source":"NewsData"})
        return articles
    except Exception as e:
        # don't spam the UI on non-critical failures
        st.warning(f"NewsData fetch error: {e}")
        return []

def fetch_from_newsapi(query, from_date, to_date):
    try:
        params = {"apiKey": NEWSAPI_API_KEY, "q": query, "from": from_date, "to": to_date, "language": "en", "pageSize": 100}
        r = requests.get(NEWSAPI_ENDPOINT, params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        articles = []
        for a in j.get("articles", []):
            articles.append({"title": a.get("title",""), "desc": a.get("description",""), "date": a.get("publishedAt",""), "url": a.get("url",""), "source":"NewsAPI"})
        return articles
    except Exception as e:
        st.warning(f"NewsAPI fetch error: {e}")
        return []

def fetch_from_gnews(query, from_date, to_date):
    try:
        params = {"token": GNEWS_API_KEY, "q": query, "from": from_date, "to": to_date, "lang": "en", "max": 100}
        r = requests.get(GNEWS_ENDPOINT, params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        articles = []
        for a in j.get("articles", []):
            articles.append({"title": a.get("title",""), "desc": a.get("description",""), "date": a.get("publishedAt",""), "url": a.get("url",""), "source":"GNews"})
        return articles
    except Exception as e:
        st.warning(f"GNews fetch error: {e}")
        return []

# -------------------------------
# STREAMLIT UI (inputs)
# -------------------------------
name = st.text_input("Enter Name / Entity / Vessel")
keywords_input = st.text_input("Enter Additional Keywords (optional, comma separated)")
keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]

st.markdown("""
> **ℹ️ Note:** This tool searches multiple news sources for adverse content and checks sanctions lists (OFAC / UK / OpenSanctions).
> The default negative keyword set includes AML, sanctions, governance, and ESG risk terms.
""")

col1, col2 = st.columns([2, 1])
with col1:
    sort_option = st.selectbox("Sort results by:", ["Newest First", "Oldest First", "Source (A-Z)"], index=0)
with col2:
    high_severity = st.checkbox("Show only high-severity (🔴) results", value=False)

if st.button("Search"):
    to_date = datetime.today().strftime("%Y-%m-%d")
    from_date = (datetime.today() - timedelta(days=365 * 7)).strftime("%Y-%m-%d")
    st.subheader("🔎 Negative News & Sanctions Matches")
    results = search_all(name, keywords, from_date, to_date, high_severity)

    if results:
        if sort_option == "Newest First":
            results.sort(key=lambda x: x.get("date", ""), reverse=True)
        elif sort_option == "Oldest First":
            results.sort(key=lambda x: x.get("date", ""))
        elif sort_option == "Source (A-Z)":
            results.sort(key=lambda x: x.get("source", "").lower())

        df = pd.DataFrame([{
            "Title": r["title"],
            "Source": r["source"],
            "Date": r["date"],
            "Severity": r["severity"],
            "Sentiment Score": r["score"],
            "Model Label": r.get("model_label", ""),
            "URL": r["url"]
        } for r in results])

        severity_colors = {"High": "red", "Medium": "orange", "Low": "#ffcc00"}
        for r in results:
            color = severity_colors.get(r["severity"], "gray")
            badge = f"<span style='color:{color}; font-weight:bold;'>({r['severity']} Severity)</span>"
            st.markdown(
                f"**{r['title']}** {badge}  \n"
                f"_Source: {r['source']} | Date: {r['date']}_  \n"
                f"{r['snippet_html']}  \n"
                f"{('[Read more](' + r['url'] + ')') if r.get('url') else ''}",
                unsafe_allow_html=True
            )
            st.write('---')

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"adverse_news_{(name or 'search')}_{datetime.today().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.write("✅ No negative news found.")