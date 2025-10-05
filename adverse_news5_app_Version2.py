import streamlit as st
import requests, re, difflib
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import pandas as pd  # For CSV export

# === Download Sentiment Model ===
nltk.download('vader_lexicon')

# === API KEYS ===
NEWSDATA_API_KEY = "pub_6f6a5665efae4a90823bc0195a8343f5"
NEWSAPI_API_KEY = "5b9afc75540c42a187ac83c8b61a165b"
GNEWS_API_KEY = "ee87c2b5d536c77dcbac6474f06497e8"

# === API Endpoints ===
NEWSDATA_ENDPOINT = "https://newsdata.io/api/1/news"
NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"
GNEWS_ENDPOINT = "https://gnews.io/api/v4/search"

# === Sanctions Sources ===
OFAC_URL = "https://sanctionslistservice.ofac.treas.gov/api/Publication"
UK_URL = "https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/1159796/UK_Sanctions_List.json"
OPENSANCTIONS_URL = "https://api.opensanctions.org/datasets/default/entities/"

# === Initialize sentiment analyzer ===
sia = SentimentIntensityAnalyzer()

# === Helper Functions ===
def fetch_ofac_list():
    try:
        resp = requests.get(OFAC_URL, timeout=30).json()
        names = []
        for entry in resp.get("SDNList", {}).get("SDNEntries", []):
            if entry.get("lastName"): names.append(entry["lastName"])
            if entry.get("firstName"): names.append(entry["firstName"])
            for aka in entry.get("akaList", {}).get("aka", []):
                if aka.get("akaName"): names.append(aka["akaName"])
        return list(set(names))
    except Exception as e:
        st.error(f"Error fetching OFAC list: {e}")
        return []

def fetch_opensanctions():
    try:
        resp = requests.get(OPENSANCTIONS_URL, timeout=30).json()
        return [e.get("properties", {}).get("name") for e in resp.get("results", []) if e.get("properties", {}).get("name")]
    except Exception as e:
        st.error(f"Error fetching OpenSanctions list: {e}")
        return []

def fetch_uk_list():
    try:
        resp = requests.get(UK_URL, timeout=30).json()
        return [x.get("Name") for x in resp if x.get("Name")]
    except Exception as e:
        st.error(f"Error fetching UK sanctions list: {e}")
        return []

def search_sanctions(name_query):
    sanctioned = fetch_ofac_list() + fetch_opensanctions() + fetch_uk_list()
    sanctioned = [s for s in sanctioned if s]
    matches = []
    for sname in sanctioned:
        ratio = difflib.SequenceMatcher(None, name_query.lower(), sname.lower()).ratio()
        if ratio >= 0.8:
            matches.append({"sanctioned_name": sname, "similarity": round(ratio, 2)})
    return sorted(matches, key=lambda x: -x["similarity"])

def fetch_from_newsdata(query, from_date, to_date):
    try:
        params = {"apikey": NEWSDATA_API_KEY, "q": query, "from_date": from_date, "to_date": to_date, "language": "en"}
        r = requests.get(NEWSDATA_ENDPOINT, params=params).json()
        articles = []
        if r and r.get("results"):
            for a in r.get("results"):
                if isinstance(a, dict):
                    articles.append({
                        "title": a.get("title", ""),
                        "desc": a.get("description", ""),
                        "date": a.get("pubDate", ""),
                        "url": a.get("link", ""),
                        "source": "NewsData"
                    })
        return articles
    except Exception as e:
        st.error(f"Error fetching from NewsData: {e}")
        return []

def fetch_from_newsapi(query, from_date, to_date):
    try:
        params = {"apiKey": NEWSAPI_API_KEY, "q": query, "from": from_date, "to": to_date, "language": "en"}
        r = requests.get(NEWSAPI_ENDPOINT, params=params).json()
        articles = []
        if r and r.get("articles"):
            for a in r.get("articles"):
                if isinstance(a, dict):
                    articles.append({
                        "title": a.get("title", ""),
                        "desc": a.get("description", ""),
                        "date": a.get("publishedAt", ""),
                        "url": a.get("url", ""),
                        "source": "NewsAPI"
                    })
        return articles
    except Exception as e:
        st.error(f"Error fetching from NewsAPI: {e}")
        return []

def fetch_from_gnews(query, from_date, to_date):
    try:
        params = {"token": GNEWS_API_KEY, "q": query, "from": from_date, "to": to_date, "lang": "en"}
        r = requests.get(GNEWS_ENDPOINT, params=params).json()
        articles = []
        if r and r.get("articles"):
            for a in r.get("articles"):
                if isinstance(a, dict):
                    articles.append({
                        "title": a.get("title", ""),
                        "desc": a.get("description", ""),
                        "date": a.get("publishedAt", ""),
                        "url": a.get("url", ""),
                        "source": "GNews"
                    })
        return articles
    except Exception as e:
        st.error(f"Error fetching from GNews: {e}")
        return []

# === Sentiment Categorization ===
def categorize_severity(score):
    """Categorize sentiment into AML severity levels."""
    if score <= -0.5:
        return "High"
    elif -0.5 < score <= -0.2:
        return "Medium"
    elif -0.2 < score < 0:
        return "Low"
    else:
        return "Not Negative"

# === Main Search Function ===
def search_all(name, keywords, from_date, to_date, high_severity=False):
    default_negative_keywords = [
        "fraud", "scam", "scandal", "ponzi", "laundering", "money laundering",
        "terrorist", "terrorism", "bribery", "corruption", "embezzlement",
        "sanction", "tax evasion", "tax fraud", "illegal", "crime", "criminal",
        "kickback", "smuggling", "forgery", "fake", "theft", "stolen",
        "misconduct", "collusion", "cartel", "black money", "suspicious",
        "investigation", "probe", "raid", "arrested", "charges", "indicted",
        "convicted", "lawsuit", "fine", "penalty", "regulatory action",
        "OFAC", "FATF", "FCPA", "terror financing", "shell company",
        "Iran", "Syria", "North Korea", "Cuba"
    ]

    all_keywords = list(set(keywords + default_negative_keywords))
    query = " ".join(([name] if name else []) + all_keywords)

    # Fetch from APIs
    articles = []
    articles += fetch_from_newsdata(query, from_date, to_date)
    articles += fetch_from_newsapi(query, from_date, to_date)
    articles += fetch_from_gnews(query, from_date, to_date)

    def color_highlight_terms(text, name, keywords):
        if name:
            text = re.sub(fr"(?i)\b({re.escape(name)})\b",
                          r"<span style='background-color: #fff176; font-weight:bold;'>\1</span>", text)
        for kw in sorted(set(keywords), key=lambda s: -len(s)):
            text = re.sub(fr"(?i)\b({re.escape(kw)})\b",
                          r"<span style='background-color: #ef9a9a; font-weight:bold;'>\1</span>", text)
        return text

    results = []
    for art in articles:
        if not isinstance(art, dict):
            continue
        text = f"{art.get('title', '')} {art.get('desc', '')}"
        score = sia.polarity_scores(text)["compound"]
        severity = categorize_severity(score)

        if severity == "Not Negative":
            continue
        if high_severity and severity != "High":
            continue

        snippet_html = color_highlight_terms(text, name, all_keywords)
        results.append({
            "title": art.get("title", ""),
            "source": art.get("source", ""),
            "date": art.get("date", ""),
            "snippet_html": snippet_html,
            "url": art.get("url", ""),
            "score": score,
            "severity": severity
        })

    return results

# === STREAMLIT UI ===
st.set_page_config(page_title="Adverse News & Sanctions Search", page_icon="📰", layout="wide")
st.title("📰 Adverse News + Sanctions Search")

name = st.text_input("Enter Name")
keywords_input = st.text_input("Enter Additional Keywords (optional, comma separated)")
keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]

st.markdown("""
> **ℹ️ Note:** This tool automatically includes negative keywords like  
> *fraud, scam, laundering, corruption, sanctions, terrorism, Iran, Syria, North Korea, Cuba,* etc.  
> You may add your own keywords if required.
""")

col1, col2 = st.columns([2, 1])
with col1:
    sort_option = st.selectbox("Sort results by:", ["Newest First", "Oldest First", "Source (A-Z)"], index=0)
with col2:
    high_severity = st.toggle("Show only high-severity (🔴) results", value=False)

if st.button("Search"):
    to_date = datetime.today().strftime("%Y-%m-%d")
    from_date = (datetime.today() - timedelta(days=365 * 7)).strftime("%Y-%m-%d")

    st.subheader("🔎 Negative News Matches")
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
                f"[Read more]({r['url']})",
                unsafe_allow_html=True
            )
            st.write('---')

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"adverse_news_{name or 'search'}_{datetime.today().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.write("✅ No negative news found.")

    if name:
        st.subheader("⚠️ Sanctions List Matches")
        ofac_matches = search_sanctions(name)
        if ofac_matches:
            for m in ofac_matches:
                st.write(f"- {m['sanctioned_name']} (similarity: {m['similarity']})")
        else:
            st.write("✅ No sanctions matches found.")