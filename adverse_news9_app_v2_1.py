import streamlit as st
import pandas as pd
import requests, re, difflib, time, os
from datetime import datetime, timedelta
from transformers import pipeline
import nltk

# -------------------------------
# CONFIGURATION / METADATA
# -------------------------------
st.set_page_config(page_title="Adverse News & Sanctions Search (v2.1)", page_icon="🛡️", layout="wide")
st.title("🛡️ Adverse News + Sanctions Search — (FinBERT default, inclusive matching)")

# -------------------------------
# MODEL SELECTION (FinBERT default)
# -------------------------------
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox(
    "Select Model:",
    ["ProsusAI/finbert", "cardiffnlp/twitter-roberta-base-sentiment-latest"],
    index=0,
    help="FinBERT default for financial/regulatory domain. RoBERTa available as alternative."
)

@st.cache_resource
def load_sentiment_model(name):
    try:
        nltk.data.find('tokenizers/punkt')
    except Exception:
        nltk.download('punkt', quiet=True)
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

# Official sanctions CSV endpoints (more stable)
OFAC_SDN_CSV = "https://www.treasury.gov/ofac/downloads/sdn.csv"
UK_SANCTIONS_CSV = "https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/1250823/UK_Sanctions_List.csv"
OPENSANCTIONS_URL = "https://api.opensanctions.org/datasets/default/entities/"

# -------------------------------
# Utilities
# -------------------------------
def normalize_name(n):
    if not n:
        return ""
    n = n.lower()
    remove_tokens = ['public joint stock company', 'pjsc', 'plc', 'llc', 'ltd', 'limited',
                     'inc', 'corp', 'corporation', 'co.', 'company', 'sa', 'gmbh', ',']
    for t in remove_tokens:
        n = n.replace(t, ' ')
    n = re.sub(r'[^a-z0-9\s]', ' ', n)
    n = re.sub(r'\s+', ' ', n).strip()
    return n

def fuzzy_match(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

# -------------------------------
# Sanctions fetchers (CSV parsing)
# -------------------------------
@st.cache_data(ttl=24*3600)
def fetch_ofac_list():
    try:
        df = pd.read_csv(OFAC_SDN_CSV, dtype=str, encoding='latin1', error_bad_lines=False)
        # SDN Name is usually in column 'SDN Name' or similar; try common column names
        possible_cols = [c for c in df.columns if 'name' in c.lower() or 'sdn' in c.lower()]
        names = []
        if possible_cols:
            for c in possible_cols:
                names += df[c].dropna().astype(str).tolist()
        else:
            # fallback: take first column
            names += df[df.columns[0]].dropna().astype(str).tolist()
        return list({n.strip() for n in names if n and str(n).strip()})
    except Exception as e:
        st.warning(f"OFAC fetch error: {e}")
        return []

@st.cache_data(ttl=24*3600)
def fetch_uk_list():
    try:
        df = pd.read_csv(UK_SANCTIONS_CSV, dtype=str, encoding='utf-8', error_bad_lines=False)
        possible_cols = [c for c in df.columns if 'name' in c.lower()]
        names = []
        if possible_cols:
            for c in possible_cols:
                names += df[c].dropna().astype(str).tolist()
        else:
            names += df[df.columns[0]].dropna().astype(str).tolist()
        return list({n.strip() for n in names if n and str(n).strip()})
    except Exception as e:
        st.warning(f"UK sanctions fetch error: {e}")
        return []

@st.cache_data(ttl=24*3600)
def fetch_opensanctions_all(limit=1000, max_items=4000):
    names = []
    try:
        offset = 0
        while True:
            url = OPENSANCTIONS_URL + f"?limit={limit}&offset={offset}"
            r = requests.get(url, timeout=30)
            if r.status_code != 200:
                break
            j = r.json()
            results = j.get('results', [])
            if not results:
                break
            for r0 in results:
                nm = r0.get('properties', {}).get('name')
                if nm:
                    names.append(nm)
            offset += limit
            if offset >= max_items:
                break
            time.sleep(0.05)
        return list({n for n in names if n})
    except Exception as e:
        st.warning(f"OpenSanctions fetch error: {e}")
        return []

# -------------------------------
# Fallback high-risk list
# -------------------------------
FALLBACK_SANCTIONS = [
    "Lukoil", "Rosneft", "Gazprom", "Sberbank", "VTB", "Alrosa", "Gazprombank", "VTB Bank", "Rosneft Oil Company"
]

# -------------------------------
# Negative keywords (concise + expanded)
# -------------------------------
KEYWORD_BUCKET = ["fraud","laundering","corruption","sanction","bribery","investigation","probe","fine","penalty"]

# builds a concise query to avoid URL length issues
def build_query(name, keywords):
    # search for name AND any of the core keywords
    kw_or = " OR ".join(keywords)
    # quote name if contains spaces
    nm = f"\"{name}\"" if " " in name.strip() else name.strip()
    query = f"{nm} AND ({kw_or})"
    return query

# -------------------------------
# News API fetchers with concise queries
# -------------------------------
def fetch_from_newsdata(name, keywords, from_date, to_date):
    try:
        q = build_query(name, keywords)
        params = {"apikey": NEWSDATA_API_KEY, "q": q, "from_date": from_date, "to_date": to_date, "language": "en"}
        r = requests.get(NEWSDATA_ENDPOINT, params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        articles = []
        for a in j.get("results", []):
            articles.append({"title": a.get("title",""), "desc": a.get("description",""), "date": a.get("pubDate",""), "url": a.get("link",""), "source":"NewsData"})
        return articles
    except Exception as e:
        st.warning(f"NewsData fetch error: {e}")
        return []

def fetch_from_newsapi(name, keywords, from_date, to_date):
    try:
        q = build_query(name, keywords)
        params = {"apiKey": NEWSAPI_API_KEY, "q": q, "from": from_date, "to": to_date, "language": "en", "pageSize": 100}
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

def fetch_from_gnews(name, keywords, from_date, to_date):
    try:
        q = build_query(name, keywords)
        params = {"token": GNEWS_API_KEY, "q": q, "from": from_date, "to": to_date, "lang": "en", "max": 100}
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
# Main search function
# -------------------------------
def search_all(name, extra_keywords, from_date, to_date, high_severity=False):
    if not name or not name.strip():
        return []
    # core keywords for concise queries
    core_keywords = KEYWORD_BUCKET
    # get news articles
    articles = []
    articles += fetch_from_newsdata(name, core_keywords, from_date, to_date)
    articles += fetch_from_newsapi(name, core_keywords, from_date, to_date)
    articles += fetch_from_gnews(name, core_keywords, from_date, to_date)
    # sanctions lists (cached)
    sanctions_names = []
    sanctions_names += fetch_ofac_list()
    sanctions_names += fetch_uk_list()
    sanctions_names += fetch_opensanctions_all(limit=1000, max_items=3000)
    sanctions_names = list({n for n in sanctions_names if n})
    sanctions_names += FALLBACK_SANCTIONS
    sanctions_names = list({n for n in sanctions_names if n})

    st.write(f"🔍 Loaded {len(sanctions_names)} sanctions entries (incl. fallback).")

    results = []
    # Sanctions fuzzy matches (inclusive threshold 0.6)
    name_norm = normalize_name(name)
    matches = []
    for s in sanctions_names:
        s_norm = normalize_name(s)
        score = fuzzy_match(name_norm, s_norm)
        if score >= 0.6:
            matches.append({"sanctioned_name": s, "similarity": round(score,2)})
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

    # Process news through transformer and keep only negative ones
    for art in articles:
        text = f"{art.get('title','')} {art.get('desc','')}"
        if not text.strip():
            continue
        try:
            res = sentiment_model(text[:1024])[0]
            label = res.get('label','') or ''
            score = float(res.get('score',0.0))
        except Exception as e:
            label = "ERROR"
            score = 0.0
        lab_lower = label.lower()
        is_negative = False
        if 'neg' in lab_lower or 'negative' in lab_lower:
            is_negative = True
        if model_choice and 'twitter-roberta' in model_choice and label.upper().startswith('LABEL_'):
            if label.upper() == 'LABEL_0':
                is_negative = True
        if not is_negative:
            continue
        if score >= 0.85:
            severity = "High"
        elif score >= 0.6:
            severity = "Medium"
        else:
            severity = "Low"
        if high_severity and severity != "High":
            continue
        # produce clickable headline only
        headline = art.get('title','').strip() or art.get('desc','').strip()
        url = art.get('url','')
        snippet_html = headline  # headlines will be shown as links in UI
        results.append({
            "title": headline,
            "source": art.get('source',''),
            "date": art.get('date',''),
            "snippet_html": snippet_html,
            "url": url,
            "score": score,
            "severity": severity,
            "model_label": label
        })
    return results

# -------------------------------
# STREAMLIT UI
# -------------------------------
name = st.text_input("Enter Name / Entity / Vessel")
extra_kw_input = st.text_input("Additional keywords (optional, comma separated)")
extra_keywords = [k.strip() for k in extra_kw_input.split(",") if k.strip()]

st.markdown("""
> **Note:** This version uses concise queries for news APIs (to avoid URL length errors) and inclusive sanctions matching (threshold 0.6).
""")

col1, col2 = st.columns([2,1])
with col1:
    sort_option = st.selectbox("Sort results by:", ["Sanctions First", "Newest First", "Oldest First", "Source (A-Z)"], index=0)
with col2:
    high_severity = st.checkbox("Show only high-severity (🔴) results", value=False)

if st.button("Search"):
    to_date = datetime.today().strftime("%Y-%m-%d")
    from_date = (datetime.today() - timedelta(days=365*7)).strftime("%Y-%m-%d")
    st.subheader("🔎 Negative News & Sanctions Matches")
    results = search_all(name, extra_keywords, from_date, to_date, high_severity)

    if results:
        if sort_option == "Newest First":
            results.sort(key=lambda x: x.get("date",""), reverse=True)
        elif sort_option == "Oldest First":
            results.sort(key=lambda x: x.get("date",""))
        elif sort_option == "Source (A-Z)":
            results.sort(key=lambda x: x.get("source","").lower())
        elif sort_option == "Sanctions First":
            results.sort(key=lambda x: (0 if x.get('source')=='SanctionsLists' else 1, x.get('date','')), reverse=False)

        # Show sanctions matches first (if any)
        for r in results:
            color = {"High":"red","Medium":"orange","Low":"#ffcc00"}.get(r.get('severity',''), 'gray')
            if r.get('source') == 'SanctionsLists':
                st.markdown(f"**{r['title']}** <span style='color:{color}; font-weight:bold;'>({r['severity']} Severity)</span><br>{r['snippet_html']}", unsafe_allow_html=True)
                st.write('---')

        st.subheader("🔎 Negative News")
        found_news = [r for r in results if r.get('source') != 'SanctionsLists']
        if not found_news:
            st.write("✅ No negative news articles found for the query.")
        else:
            for r in found_news:
                title = r.get('title','')
                url = r.get('url','')
                color = {"High":"red","Medium":"orange","Low":"#ffcc00"}.get(r.get('severity',''), 'gray')
                # clickable headline opens in new tab
                if url:
                    st.markdown(f"<a href='{url}' target='_blank' style='font-weight:bold; text-decoration:none;'>{title}</a> <span style='color:{color}; font-weight:bold;'>({r['severity']} Severity)</span><br><small>{r.get('source','')} | {r.get('date','')}</small>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span style='font-weight:bold;'>{title}</span> <span style='color:{color}; font-weight:bold;'>({r['severity']} Severity)</span><br><small>{r.get('source','')} | {r.get('date','')}</small>", unsafe_allow_html=True)
                st.write('---')

            df = pd.DataFrame([{'Title': r['title'], 'Source': r['source'], 'Date': r['date'], 'Severity': r['severity'], 'Sentiment Score': r['score'], 'URL': r['url']} for r in found_news])
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download Results as CSV", data=csv, file_name=f"adverse_news_{(name or 'search')}_{datetime.today().strftime('%Y%m%d')}.csv", mime="text/csv")
    else:
        st.write("✅ No negative news or sanctions matches found.")