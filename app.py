import os
import json
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

# ---------- Load env & init client ----------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing in .env")
if not SERPER_API_KEY:
    raise RuntimeError("SERPER_API_KEY is missing in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- Core functions ----------

def search_companies(query: str, num_results: int = 5):
    """Use Serper.dev to search the web for company websites."""
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "q": query,
        "num": num_results,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    results = []
    for item in data.get("organic", []):
        link = item.get("link")
        title = item.get("title")
        if link:
            results.append({"title": title, "link": link})
    return results


def fetch_page(url: str) -> str:
    """Fetch raw HTML for a URL."""
    try:
        resp = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""


def extract_links(html: str, base_url: str):
    """Find likely contact/about/team pages on a website."""
    soup = BeautifulSoup(html, "html.parser")
    links = set()

    for a in soup.find_all("a", href=True):
        href_raw = a["href"]
        href = href_raw.lower()
        text = (a.get_text() or "").lower()

        if any(k in href for k in ["contact", "about", "team"]) or \
           any(k in text for k in ["contact", "about", "team"]):

            # Make absolute URL if relative
            if href_raw.startswith("http"):
                links.add(href_raw)
            else:
                if href_raw.startswith("/"):
                    links.add(base_url.rstrip("/") + href_raw)
                else:
                    links.add(base_url.rstrip("/") + "/" + href_raw)

    return list(links)


# def llm_extract_company_info(website_url: str, page_text: str) -> dict:
#     """
#     Ask the LLM to extract company info from website content.
#     Returns a Python dict.
#     """
#     prompt = f"""
#     You are a data extraction assistant.

#     Given the content of a company's website, extract structured information.

#     Return **only** valid JSON with keys:
#     - company_name (string or null)
#     - website (string)
#     - emails (array of strings)
#     - phones (array of strings)
#     - address (string or null)

#     If something is missing, use null or an empty array.
#     Do not include any extra keys.

#     Website URL: {website_url}

#     Content:
#     \"\"\"{page_text[:15000]}\"\"\"
#     """

#     completion = client.chat.completions.create(
#         model="gpt-4.1-mini",
#         messages=[
#             {"role": "system", "content": "You extract structured data and respond ONLY in JSON."},
#             {"role": "user", "content": prompt},
#         ],
#         response_format={"type": "json_object"},
#     )

#     json_str = completion.choices[0].message["content"]

#     try:
#         data = json.loads(json_str)
#     except:
#         data = {
#             "company_name": None,
#             "website": website_url,
#             "emails": [],
#             "phones": [],
#             "address": None,
#             "raw": json_str
#         }

#     return data

def llm_extract_company_info(website_url: str, page_text: str) -> dict:
    """
    Ask the LLM to extract company info from website content.
    Returns a Python dict.
    """

    prompt = f"""
You are a data extraction assistant.

Extract structured contact info from the following website content.

Return ONLY valid JSON with exactly these keys:
- company_name
- website
- emails
- phones
- address

If data is missing, use null or [].

Website URL: {website_url}

Content:
\"\"\"{page_text[:15000]}\"\"\"
"""

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "You extract structured data and respond ONLY in JSON."
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
        response_format={"type": "json_object"},
    )

    # ✅ message is an object, use .content
    json_str = completion.choices[0].message.content

    try:
        data = json.loads(json_str)
    except Exception:
        # fallback if something weird comes back
        data = {
            "company_name": None,
            "website": website_url,
            "emails": [],
            "phones": [],
            "address": None,
            "raw": json_str,
        }

    # ensure keys exist
    data.setdefault("company_name", None)
    data.setdefault("website", website_url)
    data.setdefault("emails", [])
    data.setdefault("phones", [])
    data.setdefault("address", None)

    return data



def build_company_records(query: str, num_results: int = 5, sleep_sec: float = 2.0):
    """Full pipeline: search -> fetch -> extract info for each result."""
    search_results = search_companies(query, num_results=num_results)
    records = []

    for i, r in enumerate(search_results, start=1):
        url = r["link"]
        title = r["title"] or ""

        st.write(f"### {i}. {title}")
        st.write(url)

        main_html = fetch_page(url)
        if not main_html:
            st.warning("Could not fetch main page.")
            continue

        # collect content from main + contact/about pages
        content = main_html
        extra_links = extract_links(main_html, url)

        st.caption(f"Found {len(extra_links)} contact/about-like pages.")
        for idx, link in enumerate(extra_links[:5], start=1):
            st.caption(f"Fetching extra page {idx}: {link}")
            sub_html = fetch_page(link)
            content += "\n\n" + sub_html
            time.sleep(0.8)  # be nice to servers

        # LLM extraction
        with st.spinner("Extracting contact details with AI..."):
            info = llm_extract_company_info(url, content)

        # Normalize data for DataFrame
        record = {
            "query": query,
            "title": title,
            "company_name": info.get("company_name"),
            "website": info.get("website", url),
            "emails": ", ".join(info.get("emails", [])),
            "phones": ", ".join(info.get("phones", [])),
            "address": info.get("address"),
        }

        st.json(info)
        st.markdown("---")

        records.append(record)
        time.sleep(sleep_sec)

    return records


# ---------- Streamlit UI ----------

st.set_page_config(page_title="AI Company Contact Finder", layout="wide")

st.title("🏢 AI Company Contact Finder")
st.write(
    "Enter a search query like **'real estate companies in Dubai'** and the agent "
    "will try to find websites and extract emails, phone numbers, and addresses."
)

with st.sidebar:
    st.header("Settings")
    default_query = "real estate company in Dubai"
    query = st.text_input("Search query", value=default_query)
    num_results = st.slider("Number of search results to process", 1, 10, 5)
    run_button = st.button("Run Search")
    st.markdown(f"---{num_results}")
if run_button:
    if not query.strip():
        st.error("Please enter a query.")
    else:
        with st.spinner("Running agent..."):
            records = build_company_records(query, num_results=num_results)

        if not records:
            st.warning("No records extracted. Try another query or increase results.")
        else:
            df = pd.DataFrame(records)
            st.subheader("📋 Extracted Companies")
            st.dataframe(df, use_container_width=True)

            # CSV download
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download results as CSV",
                data=csv,
                file_name="company_contacts.csv",
                mime="text/csv",
            )
else:
    st.info("Configure your query in the sidebar and click **Run Search** to start.")
