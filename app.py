import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
import sqlite3
from datetime import datetime
from urllib.parse import urlparse
from werkzeug.security import generate_password_hash, check_password_hash
import threading
import logging
import re

# ---------- Load env & init client ----------
DB_PATH = "company_contacts.db"
load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Default admin credentials (can be set in .env)
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "admin123")

if not SERPER_API_KEY:
    raise RuntimeError("SERPER_API_KEY is missing in .env")

# ---------- Logging Setup ----------
logging.basicConfig(
    filename='app_search.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------- Database initialization ----------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # legacy contacts table (kept for safety, not used actively)
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
    # new simplified links table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS website_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            website TEXT,
            website_normalized TEXT UNIQUE,
            category TEXT,
            location TEXT,
            page_no INTEGER,
            created_at TEXT
        )
        """
    )
    # users table
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
    # categories table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE
        )
        """
    )
    # locations table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS locations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE
        )
        """
    )
    # app_settings table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS app_settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    # search_queue table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS search_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT,
            location TEXT,
            status TEXT DEFAULT 'pending',
            created_at TEXT
        )
        """
    )
    
    # Initialize default settings if not exists
    cur.execute("INSERT OR IGNORE INTO app_settings (key, value) VALUES ('interval_minutes', '20')")
    cur.execute("INSERT OR IGNORE INTO app_settings (key, value) VALUES ('search_pages', '3')")
    cur.execute("INSERT OR IGNORE INTO app_settings (key, value) VALUES ('is_running', 'false')")
    
    conn.commit()
    conn.close()

def execute_db(query, args=()):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute(query, args)
        conn.commit()
        return cur
    except Exception as e:
        logger.error(f"DB Error: {e}")
        return None
    finally:
        conn.close()

def fetch_db(query, args=()):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(query, args)
    rows = cur.fetchall()
    conn.close()
    return rows

def fetch_db_one(query, args=()):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(query, args)
    row = cur.fetchone()
    conn.close()
    return row

def get_setting(key, default=""):
    row = fetch_db_one("SELECT value FROM app_settings WHERE key = ?", (key,))
    return row[0] if row else default

def update_setting(key, value):
    execute_db("UPDATE app_settings SET value = ? WHERE key = ?", (str(value), key))

# --- Authentication Helpers ---
def create_user(username: str, password: str) -> dict:
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
    row = fetch_db_one("SELECT id, username, password_hash FROM users WHERE username = ?", (username,))
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
    existing = get_user_by_username(ADMIN_USER)
    if existing:
        return existing
    return create_user(ADMIN_USER, ADMIN_PASS)

# ---------- Lightweight Link Extraction ----------
def normalize_url(url: str) -> str:
    if not url:
        return ""
    try:
        url_strip = url.strip()
        if not url_strip.startswith("http"):
            url_strip = "https://" + url_strip
        parsed = urlparse(url_strip)
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return url.strip().lower()

def search_companies_paged(query: str, target_pages: int = 3, ui_mode=True):
    """Fetches purely links and records which Serper page they came from."""
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    results = []
    page_size = 10
    
    for page in range(1, target_pages + 1):
        payload = {"q": query, "num": page_size, "page": page}
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            organic = data.get("organic", []) or []
            if not organic:
                break
            for item in organic:
                link = item.get("link")
                if link:
                    results.append({"link": link, "page_no": page})
        except Exception as e:
            msg = f"Search API error on page {page}: {e}"
            if ui_mode: st.error(msg)
            logger.error(msg)
            break
    return results

def build_website_links(category: str, location: str, target_pages: int = 3, ui_mode: bool = True):
    query = f"{category} in {location}"
    if ui_mode: st.markdown(f"## Searching: '{query}' across {target_pages} pages")
    
    search_results = search_companies_paged(query, target_pages=target_pages, ui_mode=ui_mode)
    if ui_mode: st.write(f"Got {len(search_results)} links from API")
    
    records = []
    for r in search_results:
        records.append({
            "website": r["link"],
            "category": category,
            "location": location,
            "page_no": r["page_no"]
        })
    return records

def save_links_to_db(records):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    inserted = 0
    skipped = 0
    for r in records:
        website = r.get("website") or ""
        normalized = normalize_url(website)
        website = normalized # Only store domain name
        category = r.get("category") or ""
        location = r.get("location") or ""
        page_no = r.get("page_no") or 1
        
        try:
            cur.execute(
                """
                INSERT OR IGNORE INTO website_links
                (website, website_normalized, category, location, page_no, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (website, normalized, category, location, page_no, datetime.utcnow().isoformat()),
            )
            if cur.rowcount == 0:
                skipped += 1
            else:
                inserted += 1
        except Exception as e:
            logger.error(f"DB insert error: {e}")
            skipped += 1
    conn.commit()
    conn.close()
    return inserted, skipped

def load_links_from_db():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT id, website, category, location, page_no, created_at FROM website_links ORDER BY id DESC", conn)
    conn.close()
    return df

# ---------- Background Thread Logic ----------
def background_search_worker():
    while True:
        try:
            is_running = get_setting('is_running') == 'true'
            if not is_running:
                time.sleep(10)
                continue
                
            # Check for pending items
            next_task = fetch_db_one("SELECT id, category, location FROM search_queue WHERE status = 'pending' ORDER BY id ASC LIMIT 1")
            
            if next_task:
                task_id, category, location = next_task
                
                # Mark as processing
                execute_db("UPDATE search_queue SET status = 'processing' WHERE id = ?", (task_id,))
                logger.info(f"Background worker starting: {category} in {location}")
                
                # Get settings
                pages = int(get_setting('search_pages', '3'))
                
                records = build_website_links(category, location, target_pages=pages, ui_mode=False)
                
                if records:
                    inserted, skipped = save_links_to_db(records)
                    logger.info(f"Completed {category} in {location}. Inserted: {inserted}, Skipped: {skipped}")
                else:
                    logger.info(f"'{category} in {location}' returned no valid records.")
                    
                # Mark as completed
                execute_db("UPDATE search_queue SET status = 'completed' WHERE id = ?", (task_id,))
                
                # Check if queue is empty now
                remaining = fetch_db_one("SELECT COUNT(*) FROM search_queue WHERE status = 'pending'")
                if remaining and remaining[0] == 0:
                    logger.info("All category searches are completed. Stopping process.")
                    update_setting('is_running', 'false')
                    
                # Sleep for configured interval
                interval = int(get_setting('interval_minutes', '20'))
                logger.info(f"Sleeping for {interval} minutes before next task.")
                time.sleep(interval * 60)
            else:
                # No tasks but running? Stop it.
                update_setting('is_running', 'false')
                time.sleep(10)
                
        except Exception as e:
            logger.error(f"Background worker error: {e}")
            time.sleep(60)

# Start background thread only once
if 'bg_thread_started' not in st.session_state:
    thread = threading.Thread(target=background_search_worker, daemon=True)
    thread.start()
    st.session_state['bg_thread_started'] = True

# ---------- App UI with Auth ----------
st.set_page_config(page_title="Website Link Finder", layout="wide")
init_db()
ensure_default_admin()

# ---------- Sidebar: Authentication & Navigation ----------
with st.sidebar:
    st.header("🔐 Account")
    if "user_info" in st.session_state and st.session_state.get("user_info"):
        user = st.session_state["user_info"]
        st.write(f"Signed in as **{user.get('username')}**")
        if st.button("Logout"):
            st.session_state.pop("user_info", None)
            st.rerun()
            
        st.markdown("---")
        st.header("Navigation")
        page = st.radio("Go to", ["Manual Search", "Automated Search", "Search List", "Logs"])
    else:
        page = "Login"
        auth_tab = st.radio("Action", ["Login", "Register"])
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
    st.title("Website Link Finder")
    st.write("Please log in (sidebar) to access the features.")
    st.stop()

if page == "Manual Search":
    st.title("🏢 Manual Search")
    st.write("Run a one-off query and instantly extract the website links.")
    
    col1, col2 = st.columns(2)
    with col1:
        manual_cat = st.text_input("Category (e.g. SEO Agency)", value="SEO Agency")
    with col2:
        manual_loc = st.text_input("Location (e.g. Dubai)", value="Dubai")
        
    num_pages = st.slider("Number of pages to search (10 results per page)", 1, 10, 3)
    run_button = st.button("Run Manual Search")
    
    if run_button:
        if not manual_cat.strip() or not manual_loc.strip():
            st.error("Please enter both a Category and Location.")
        else:
            with st.spinner("Extracting links..."):
                records = build_website_links(manual_cat, manual_loc, target_pages=num_pages, ui_mode=True)
            if not records:
                st.warning("No links extracted. Try another query or increase results.")
            else:
                df = pd.DataFrame(records)
                st.session_state["records_df"] = df
                inserted, skipped = save_links_to_db(records)
                st.success(f"Saved to database. Inserted: {inserted} new, skipped (duplicates): {skipped}")
                
    if "records_df" in st.session_state:
        st.subheader("📋 Last Manual Search Results")
        st.dataframe(st.session_state["records_df"], use_container_width=True)

elif page == "Automated Search":
    st.title("⚙️ Automated Background Search")
    
    # 1. Categories
    st.subheader("1. Categories")
    col1, col2 = st.columns(2)
    with col1:
        new_cats = st.text_area("Add Categories (comma or newline separated)")
        if st.button("Add Categories"):
            if new_cats:
                parts = re.split(r'[,\n]', new_cats)
                for part in parts:
                    val = part.strip()
                    if val:
                        execute_db("INSERT OR IGNORE INTO categories (name) VALUES (?)", (val,))
                st.rerun()
    with col2:
        cats = fetch_db("SELECT id, name FROM categories")
        if cats:
            cat_options = {cname: cid for cid, cname in cats}
            selected_cat_name = st.selectbox("Select Category to Edit/Delete", list(cat_options.keys()))
            if selected_cat_name:
                selected_cat_id = cat_options[selected_cat_name]
                edit_cat_val = st.text_input("Edit Category", value=selected_cat_name, key="edit_cat")
                col2_1, col2_2 = st.columns(2)
                if col2_1.button("Update Category"):
                    if edit_cat_val and edit_cat_val.strip() != selected_cat_name:
                        execute_db("UPDATE categories SET name=? WHERE id=?", (edit_cat_val.strip(), selected_cat_id))
                        st.rerun()
                if col2_2.button("❌ Delete Category"):
                    execute_db("DELETE FROM categories WHERE id=?", (selected_cat_id,))
                    st.rerun()
        else:
            st.write("No categories yet.")

    st.markdown("---")
    # 2. Locations
    st.subheader("2. Locations")
    col3, col4 = st.columns(2)
    with col3:
        new_locs = st.text_area("Add Locations (comma or newline separated)")
        if st.button("Add Locations"):
            if new_locs:
                parts = re.split(r'[,\n]', new_locs)
                for part in parts:
                    val = part.strip()
                    if val:
                        execute_db("INSERT OR IGNORE INTO locations (name) VALUES (?)", (val,))
                st.rerun()
    with col4:
        locs = fetch_db("SELECT id, name FROM locations")
        if locs:
            loc_options = {lname: lid for lid, lname in locs}
            selected_loc_name = st.selectbox("Select Location to Edit/Delete", list(loc_options.keys()))
            if selected_loc_name:
                selected_loc_id = loc_options[selected_loc_name]
                edit_loc_val = st.text_input("Edit Location", value=selected_loc_name, key="edit_loc")
                col4_1, col4_2 = st.columns(2)
                if col4_1.button("Update Location"):
                    if edit_loc_val and edit_loc_val.strip() != selected_loc_name:
                        execute_db("UPDATE locations SET name=? WHERE id=?", (edit_loc_val.strip(), selected_loc_id))
                        st.rerun()
                if col4_2.button("❌ Delete Location"):
                    execute_db("DELETE FROM locations WHERE id=?", (selected_loc_id,))
                    st.rerun()
        else:
            st.write("No locations yet.")

    st.markdown("---")
    # 3. Settings & Queue Control
    st.subheader("3. Settings & Queue Control")
    current_interval = int(get_setting("interval_minutes", "20"))
    current_pages = int(get_setting("search_pages", "3"))
    
    interval = st.number_input("Interval between searches (minutes)", min_value=1, value=current_interval)
    pages = st.number_input("Pages to search per category (1 page = 10 results)", min_value=1, max_value=20, value=current_pages)
    
    if st.button("Save Settings"):
        update_setting("interval_minutes", interval)
        update_setting("search_pages", pages)
        st.success("Settings saved!")
        
    st.markdown("---")
    
    st.subheader("Queue Status")
    
    col_q1, col_q2 = st.columns(2)
    with col_q1:
        if st.button("Generate Queue from Categories & Locations"):
            cats = fetch_db("SELECT name FROM categories")
            locs = fetch_db("SELECT name FROM locations")
            if not cats or not locs:
                st.error("Please add at least one category and one location.")
            else:
                execute_db("DELETE FROM search_queue WHERE status = 'pending'") # clear old pending
                for c in cats:
                    for l in locs:
                        execute_db("INSERT INTO search_queue (category, location, created_at) VALUES (?, ?, ?)", (c[0], l[0], datetime.utcnow().isoformat()))
                st.success("Queue generated successfully!")
                st.rerun()
    with col_q2:
        if st.button("Clear Queue"):
            execute_db("DELETE FROM search_queue")
            st.success("Queue cleared.")
            st.rerun()

    queue_data = fetch_db("SELECT id, category, location, status FROM search_queue ORDER BY id")
    if queue_data:
        df_queue = pd.DataFrame(queue_data, columns=["ID", "Category", "Location", "Status"])
        st.dataframe(df_queue, use_container_width=True)
    else:
        st.info("Queue is empty.")

    is_running = get_setting("is_running") == "true"
    if is_running:
        st.success("🟢 Automated Search is currently RUNNING")
        if st.button("Stop Automated Search"):
            update_setting("is_running", "false")
            st.rerun()
    else:
        st.warning("🔴 Automated Search is STOPPED")
        if st.button("Start Automated Search"):
            update_setting("is_running", "true")
            st.rerun()

elif page == "Search List":
    st.title("📚 Search List")
    st.write("All automatically and manually extracted website links.")
    
    db_df = load_links_from_db()
    
    if db_df.empty:
        st.info("No records found in database.")
    else:
        # Rename columns to look nicer in UI
        display_df = db_df.rename(columns={
            "id": "ID",
            "website": "Website Link",
            "category": "Category",
            "location": "Location",
            "page_no": "Search Page Number",
            "created_at": "Insert Data Time"
        })
        
        total_rows = len(display_df)
        page_size = st.number_input("Rows per page", min_value=5, max_value=500, value=50, step=10)
        total_pages = (total_rows + page_size - 1) // page_size
        page_no = st.number_input("Page view number", min_value=1, max_value=max(1, total_pages), value=1, step=1)
        start_idx = (page_no - 1) * page_size
        end_idx = start_idx + page_size
        
        st.caption(f"Showing {start_idx + 1}–{min(end_idx, total_rows)} of {total_rows} records. You can edit cells directly in the table below and click 'Save Changes'.")
        
        current_page_df = display_df.iloc[start_idx:end_idx]
        
        edited_df = st.data_editor(
            current_page_df, 
            use_container_width=True,
            disabled=["ID", "Search Page Number", "Insert Data Time"],
            hide_index=True
        )
        
        if st.button("Save Changes"):
            changes_made = 0
            for i in range(len(current_page_df)):
                orig = current_page_df.iloc[i]
                new = edited_df.iloc[i]
                if (orig["Website Link"] != new["Website Link"] or 
                    orig["Category"] != new["Category"] or 
                    orig["Location"] != new["Location"]):
                    
                    execute_db(
                        "UPDATE website_links SET website=?, website_normalized=?, category=?, location=? WHERE id=?",
                        (
                            new["Website Link"],
                            normalize_url(new["Website Link"]),
                            new["Category"],
                            new["Location"],
                            int(new["ID"])
                        )
                    )
                    changes_made += 1
            if changes_made > 0:
                st.success(f"Saved {changes_made} changes!")
                st.rerun()
            else:
                st.info("No changes detected.")
        
        csv = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(label="⬇️ Download ALL results as CSV", data=csv, file_name="website_links.csv", mime="text/csv")

elif page == "Logs":
    st.title("📝 System Logs")
    if os.path.exists('app_search.log'):
        with open('app_search.log', 'r') as f:
            logs = f.readlines()
        st.text_area("app_search.log", "".join(logs[-100:]), height=600) # show last 100 lines
        if st.button("Clear Logs"):
            open('app_search.log', 'w').close()
            st.rerun()
    else:
        st.info("Log file not found or empty.")
