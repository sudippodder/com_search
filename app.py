# updated_app_with_auth.py
import os
from huggingface_hub import InferenceClient
import json
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
#from openai import OpenAI
import streamlit as st
import re
import sqlite3
from datetime import datetime
from urllib.parse import urlparse, unquote
from werkzeug.security import generate_password_hash, check_password_hash

# ---------- Load env & init client ----------
DB_PATH = "company_contacts.db"

load_dotenv()

#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Default admin credentials (can be set in .env)
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "admin123")

# if not OPENAI_API_KEY:
#     raise RuntimeError("OPENAI_API_KEY is missing in .env")
if not SERPER_API_KEY:
    raise RuntimeError("SERPER_API_KEY is missing in .env")

API_TOKEN = os.getenv("HF_TOKEN")
#TEXT_CLASSIFICATION_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
TEXT_CLASSIFICATION_MODEL = "HuggingFaceTB/SmolLM3-3B"
client = InferenceClient(model=TEXT_CLASSIFICATION_MODEL, token=API_TOKEN)

# ---------- Database initialization ----------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # contacts table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS company_contacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            title TEXT,
            company_name TEXT,
            website TEXT,
            website_normalized TEXT UNIQUE,
            emails TEXT,
            phones TEXT,
            address TEXT,
            created_at TEXT
        )
        """
    )
    # users table (for authentication)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash TEXT,
            created_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()

def create_user(username: str, password: str) -> dict:
    """
    Create a user with hashed password. Returns dict with id/username on success or error.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        password_hash = generate_password_hash(password)
        cur.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, password_hash, datetime.utcnow().isoformat())
        )
        conn.commit()
        user_id = cur.lastrowid
        return {"id": user_id, "username": username}
    except sqlite3.IntegrityError:
        return {"error": "username_exists"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()

def get_user_by_username(username: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, username, password_hash FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {"id": row[0], "username": row[1], "password_hash": row[2]}

def verify_user(username: str, password: str):
    user = get_user_by_username(username)
    if not user:
        return {"ok": False, "error": "user_not_found"}
    if check_password_hash(user["password_hash"], password):
        return {"ok": True, "id": user["id"], "username": user["username"]}
    return {"ok": False, "error": "invalid_password"}

def ensure_default_admin():
    """
    Create default admin user if not present, using ADMIN_USER / ADMIN_PASS env vars.
    """
    existing = get_user_by_username(ADMIN_USER)
    if existing:
        return existing
    res = create_user(ADMIN_USER, ADMIN_PASS)
    return res

# ---------- existing helper functions (unchanged) ----------
def search_companies(query: str, total_results: int = 50, page_size: int = 10):
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    results = []
    page = 1
    while len(results) < total_results:
        payload = {"q": query, "num": page_size, "page": page}
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        organic = data.get("organic", []) or []
        if not organic:
            break
        for item in organic:
            link = item.get("link")
            title = item.get("title")
            if link:
                results.append({"title": title, "link": link})
                if len(results) >= total_results:
                    break
        page += 1
    return results

def extract_contacts_from_html(html: str):
    soup = BeautifulSoup(html, "html.parser")
    emails = set()
    phones = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.lower().startswith("mailto:"):
            unquoted_href = unquote(href)
            email_with_params = unquoted_href.split("mailto:", 1)[1].split("?", 1)[0]
            email = email_with_params.strip()
            if email and "@" in email:
                emails.add(email.lower())
        if href.lower().startswith("tel:"):
            phone = href.split("tel:", 1)[1].strip()
            phone = unquote(phone)
            phone = phone.replace("%20", " ")
            if phone:
                phones.add(phone)
    full_text = soup.get_text(separator=" ", strip=True)
    full_text = re.sub(r"\s+", " ", full_text)
    email_pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    phone_pattern = r"\+?\d[\d\s\-()]{6,}"
    for email in re.findall(email_pattern, full_text):
        emails.add(email.strip().lower())
    for phone in re.findall(phone_pattern, full_text):
        phones.add(phone.strip())
    clean_emails = sorted(list({e.replace(" ", "") for e in emails if e}))
    clean_phones = sorted(list({p for p in phones if p}))
    return clean_emails, clean_phones

def fetch_page(url: str) -> str:
    try:
        resp = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

def extract_links(html: str, base_url: str):
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href_raw = a["href"]
        href = href_raw.lower()
        text = (a.get_text() or "").lower()
        if any(k in href for k in ["contact", "about", "team"]) or any(k in text for k in ["contact", "about", "team"]):
            if href_raw.startswith("http"):
                links.add(href_raw)
            else:
                if href_raw.startswith("/"):
                    links.add(base_url.rstrip("/") + href_raw)
                else:
                    links.add(base_url.rstrip("/") + "/" + href_raw)
    return list(links)

def llm_extract_company_info(website_url: str, page_text: str) -> dict:
    prompt = f"""
You are a data extraction assistant.

Extract structured contact info from the following HTML of a company's website.
HTML may contain Elementor/WordPress blocks, <ul><li> lists, <a href="mailto:..."> and <a href="tel:..."> tags.

Return ONLY valid JSON with exactly these keys:
- company_name
- website
- emails
- phones
- address

If data is missing, use null or [].

Website URL: {website_url}

HTML:
\"\"\"{page_text[:15000]}\"\"\" 
"""
    completion = client.chat.completions.create(
        model=TEXT_CLASSIFICATION_MODEL,
        messages=[
            {"role": "system", "content": "You extract structured data and respond ONLY in JSON."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    json_str = completion.choices[0].message.content
    try:
        data = json.loads(json_str)
    except Exception:
        data = {
            "company_name": None,
            "website": website_url,
            "emails": [],
            "phones": [],
            "address": None,
            "raw": json_str,
        }
    data.setdefault("company_name", None)
    data.setdefault("website", website_url)
    data.setdefault("emails", [])
    if data["emails"] is None:
        data["emails"] = []
    data.setdefault("phones", [])
    if data["phones"] is None:
        data["phones"] = []
    data.setdefault("address", None)
    return data

def search_companies_paged(query: str, total_results: int = 30, page_size: int = 10):
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    results = []
    page = 1
    while len(results) < total_results:
        payload = {"q": query, "num": page_size, "page": page}
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        organic = data.get("organic", []) or []
        if not organic:
            break
        for item in organic:
            link = item.get("link")
            title = item.get("title")
            if link:
                results.append({"title": title, "link": link})
                if len(results) >= total_results:
                    break
        page += 1
    return results

def normalize_url(url: str) -> str:
    if not url:
        return ""
    try:
        parsed = urlparse(url.strip())
        scheme = parsed.scheme.lower() or "https"
        netloc = parsed.netloc.lower()
        path = parsed.path.rstrip("/")
        normalized = f"{scheme}://{netloc}{path}"
        return normalized
    except Exception:
        return url.strip().lower()

def save_records_to_db(records):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    inserted = 0
    skipped = 0
    for r in records:
        website = r.get("website") or ""
        normalized = normalize_url(website)
        query = r.get("query") or ""
        title = r.get("title") or ""
        company_name = r.get("company_name") or ""
        emails = r.get("emails") or ""
        phones = r.get("phones") or ""
        address = r.get("address") or ""
        try:
            cur.execute(
                """
                INSERT OR IGNORE INTO company_contacts
                (query, title, company_name, website, website_normalized, emails, phones, address, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    query,
                    title,
                    company_name,
                    website,
                    normalized,
                    emails,
                    phones,
                    address,
                    datetime.utcnow().isoformat(),
                ),
            )
            if cur.rowcount == 0:
                skipped += 1
            else:
                inserted += 1
        except Exception as e:
            print("DB insert error:", e)
            skipped += 1
    conn.commit()
    conn.close()
    return inserted, skipped

def load_all_from_db():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM company_contacts ORDER BY id DESC", conn)
    conn.close()
    return df

def build_company_records(query: str, num_results: int = 5, sleep_sec: float = 2.0):
    st.markdown(f"## Searching for: {num_results} companies matching '{query}'")
    search_results = search_companies_paged(query, total_results=num_results)
    st.write(f"Got {len(search_results)} search results from API")
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
        content = main_html
        extra_links = extract_links(main_html, url)
        st.caption(f"Found {len(extra_links)} contact/about-like pages.")
        for idx, link in enumerate(extra_links[:5], start=1):
            st.caption(f"Fetching extra page {idx}: {link}")
            sub_html = fetch_page(link)
            content += "\n\n" + sub_html
            time.sleep(0.8)
        emails_bs, phones_bs = extract_contacts_from_html(content)
        with st.spinner("Extracting contact details with AI..."):
            info = llm_extract_company_info(url, content)
        # Normalize LLM results and merge
        llm_emails_list = info.get("emails")
        llm_emails_list = llm_emails_list if isinstance(llm_emails_list, list) else []
        llm_emails = [str(e) for e in llm_emails_list if e is not None]
        info_emails = set(llm_emails)
        llm_phones_list = info.get("phones")
        llm_phones_list = llm_phones_list if isinstance(llm_phones_list, list) else []
        llm_phones = [str(p) for p in llm_phones_list if p is not None]
        info_phones = set(llm_phones)
        merged_emails = sorted(info_emails.union(emails_bs))
        merged_phones = sorted(info_phones.union(phones_bs))
        info["emails"] = merged_emails
        info["phones"] = merged_phones
        
         # === FIX APPLIED HERE: Standardize 'address' to a string ===
        raw_address = info.get("address")
        final_address = ""
        if isinstance(raw_address, list):
            # If the LLM returned a list of addresses, join them into one string
            final_address = " / ".join([str(a) for a in raw_address if a])
        elif raw_address is not None:
            # If it's a string or other scalar, convert to string
            final_address = str(raw_address)
        # ==========================================================
        
        
        record = {
            "query": query,
            "title": title,
            "company_name": info.get("company_name"),
            "website": info.get("website", url),
            "emails": ", ".join(info.get("emails", [])),
            "phones": ", ".join(info.get("phones", [])),
            "address": final_address,
        }
        st.json(info)
        st.markdown("---")
        records.append(record)
        time.sleep(sleep_sec)
    return records

# ---------- App UI with Auth ----------
st.set_page_config(page_title="AI Company Contact Finder", layout="wide")
init_db()
ensure_default_admin()

# ---------- Sidebar: Authentication ----------
with st.sidebar:
    st.header("🔐 Account")
    if "user_info" in st.session_state and st.session_state.get("user_info"):
        user = st.session_state["user_info"]
        st.write(f"Signed in as **{user.get('username')}**")
        if st.button("Logout"):
            st.session_state.pop("user_info", None)
            st.rerun()
    else:
        auth_tab = st.radio("Action", ("Login"))
        if auth_tab == "Login":
            login_user = st.text_input("Username", key="login_user")
            login_pass = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login"):
                res = verify_user(login_user, login_pass)
                if res.get("ok"):
                    st.session_state["user_info"] = {"id": res["id"], "username": res["username"]}
                    st.success(f"Welcome, {res['username']}!")
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
        else:
            new_user = st.text_input("New username", key="new_user")
            new_pass = st.text_input("New password", type="password", key="new_pass")
            confirm_pass = st.text_input("Confirm password", type="password", key="confirm_pass")
            if st.button("Register"):
                if not new_user or not new_pass:
                    st.error("Fill both username and password.")
                elif new_pass != confirm_pass:
                    st.error("Passwords do not match.")
                else:
                    created = create_user(new_user, new_pass)
                    if created.get("error") == "username_exists":
                        st.error("Username already exists. Pick another.")
                    elif created.get("id"):
                        st.success("Account created. You can now login.")
                    else:
                        st.error(f"Could not create account: {created.get('error')}")

# ---------- Main content: protected ----------
if "user_info" not in st.session_state or not st.session_state.get("user_info"):
    st.title("AI Company Contact Finder")
    st.write(
        "Please log in (sidebar) to access the search and extraction features."
    )
    st.stop()

# From here on the user is authenticated
st.title("🏢 AI Company Contact Finder")
st.write(
    "Enter a search query like **'real estate companies in Dubai'** and the agent "
    "will try to find websites and extract emails, phone numbers, and addresses."
)

with st.sidebar:
    st.header("Settings")
    default_query = "real estate company in Dubai"
    query = st.text_input("Search query", value=default_query)
    num_results = st.sidebar.slider("Number of search results to process", 1, 100, 20)
    run_button = st.button("Run Search")
    st.markdown(f"---{num_results}")

if run_button:
    if not query.strip():
        st.error("Please enter a query.")
    else:
        with st.spinner("Running agent..."):
            records = build_company_records(query, num_results=num_results,sleep_sec=2.0)
        if not records:
            st.warning("No records extracted. Try another query or increase results.")
        else:
            df = pd.DataFrame(records)
            st.session_state["records_df"] = df
            inserted, skipped = save_records_to_db(records)
            st.success(f"Saved to database. Inserted: {inserted}, skipped (duplicates): {skipped}")
else:
    st.info("Configure your query in the sidebar and click **Run Search** to start.")

# ---------- Pagination display ----------
if "records_df" in st.session_state:
    df = st.session_state["records_df"]
    st.subheader("📋 Extracted Companies (Paginated)")
    total_rows = len(df)
    page_size = st.sidebar.number_input("Rows per page", min_value=5, max_value=100, value=10, step=5)
    total_pages = (total_rows + page_size - 1) // page_size
    page = st.sidebar.number_input("Page number", min_value=1, max_value=max(1, total_pages), value=1, step=1)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    st.caption(f"Showing {start_idx + 1}–{min(end_idx, total_rows)} of {total_rows} records")
    st.dataframe(df.iloc[start_idx:end_idx], use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label="⬇️ Download ALL results as CSV", data=csv, file_name="company_contacts.csv", mime="text/csv")
else:
    st.info("Run a search to see results here.")

with st.expander("📚 View data saved in DB"):
    db_df = load_all_from_db()
    st.dataframe(db_df, use_container_width=True)
