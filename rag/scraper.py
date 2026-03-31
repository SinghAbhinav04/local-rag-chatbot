"""
Web scraper — fetch a URL, strip boilerplate, chunk and embed
the content into an existing ChromaDB collection.
"""

import re
import urllib.parse
import os
import json

from curl_cffi import requests as cffi_requests
from bs4 import BeautifulSoup
import chromadb
import time

from rag.config  import (SCRAPED_DOCS_FOLDER, REDDIT_JSON_FOLDER, 
                        SCRAPED_DATA_FOLDER, X_SCRAPED_FOLDER, INSTA_SCRAPED_FOLDER)
from rag.console import console
from rag.chunking import chunk_text
from rag.vectordb import get_embedding
from rag.loaders  import load_file


def parse_reddit_json(data) -> str:
    """
    Recursively extract post text and comments from Reddit's listing JSON.
    """
    out = []
    
    # Reddit JSON is usually a list: [post_listing, comments_listing]
    if isinstance(data, list):
        for item in data:
            out.append(parse_reddit_json(item))
        return "\n\n".join(filter(None, out))

    # If it's a listing, look at children
    if isinstance(data, dict):
        kind = data.get("kind")
        obj_data = data.get("data", {})
        
        # Post (t3) or Comment (t1)
        if kind in ["t1", "t3"]:
            title = obj_data.get("title", "")
            body  = obj_data.get("selftext") or obj_data.get("body", "")
            author = obj_data.get("author", "[unknown]")
            
            content = []
            if title: content.append(f"Post Title: {title}")
            if author: content.append(f"Author: {author}")
            if body: content.append(body)
            out.append("\n".join(content))
            
            # Handle replies
            replies = obj_data.get("replies")
            if replies:
                out.append(parse_reddit_json(replies))
                
        # Listing or other container
        children = obj_data.get("children")
        if children:
            for child in children:
                out.append(parse_reddit_json(child))
                
    return "\n\n".join(filter(None, out))


def get_insta_cookies() -> dict:
    """Read www.instagram.com_cookies.txt and return a dictionary for curl_cffi."""
    cookies = {}
    path = "www.instagram.com_cookies.txt"
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split("\t")
                if len(parts) >= 7:
                    name = parts[5]
                    val = parts[6]
                    if val.startswith('"') and val.endswith('"'):
                        val = val[1:-1]
                    cookies[name] = val
    except Exception as e:
        console.print(f"  [error]Failed to parse {path}: {e}[/]")
    return cookies


def scrape_insta(url: str, folder: str = None) -> tuple[str, str]:
    """Specialized scraper for Instagram using cookies and web_profile_info API."""
    if folder is None:
        folder = INSTA_SCRAPED_FOLDER
    
    parsed = urllib.parse.urlparse(url)
    # Extract username from URL: instagram.com/username/ or instagram.com/username
    path_parts = [p for p in parsed.path.split("/") if p]
    if not path_parts:
        return "Invalid Instagram URL.", "instagram.com"
    
    target_username = path_parts[0]
    
    cookies = get_insta_cookies()
    if not cookies:
        console.print(f"  [error]No cookies found in www.instagram.com_cookies.txt. Instagram scraping will likely fail.[/]")

    session = cffi_requests.Session()
    session.cookies.update(cookies)
    
    # Headers to mimic a browser
    headers = {
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
        "x-ig-app-id": "936619743392459", # Standard web app ID
        "x-requested-with": "XMLHttpRequest",
        "referer": f"https://www.instagram.com/{target_username}/",
        "x-csrftoken": cookies.get("csrftoken", "")
    }

    try:
        api_url = f"https://www.instagram.com/api/v1/users/web_profile_info/?username={target_username}"
        console.print(f"  [system]Instagram:[/] Fetching profile info for @{target_username}...")
        
        resp = session.get(api_url, headers=headers, impersonate="chrome110", timeout=15)
        
        if resp.status_code != 200:
            console.print(f"  [error]Instagram API returned status {resp.status_code}. Profile might be private or cookies expired.[/]")
            return f"Failed to fetch Instagram profile for @{target_username}.", "instagram.com"

        data = resp.json()
        user = data.get("data", {}).get("user", {})
        
        if not user:
            return f"Could not find user data for @{target_username}.", "instagram.com"

        output = []
        output.append(f"Profile: {user.get('full_name')} (@{user.get('username')})")
        if user.get('category_name'):
            output.append(f"Category: {user.get('category_name')}")
        output.append(f"Bio: {user.get('biography')}")
        
        # Contact Info
        email = user.get("business_email")
        phone = user.get("business_phone_number")
        if email: output.append(f"Email: {email}")
        if phone: output.append(f"Phone: {phone}")
        
        # Address Info
        addr_json = user.get("business_address_json")
        if addr_json:
            try:
                addr_data = json.loads(addr_json)
                street = addr_data.get("street_address")
                city = addr_data.get("city_name")
                zip_c = addr_data.get("zip_code")
                addr_parts = [p for p in [street, city, zip_c] if p]
                if addr_parts:
                    output.append(f"Address: {', '.join(addr_parts)}")
            except:
                pass

        if user.get("external_url"):
            output.append(f"External URL: {user.get('external_url')}")
        output.append(f"Stats: {user.get('edge_followed_by', {}).get('count')} followers, {user.get('edge_follow', {}).get('count')} following")
        output.append("-" * 30)

        # Extract Posts using feed endpoint for better reliability on large accounts
        user_id = user.get("id")
        feed_url = f"https://www.instagram.com/api/v1/feed/user/{user_id}/"
        console.print(f"  [system]Instagram:[/] Fetching latest feed for UID {user_id}...")
        
        feed_resp = session.get(feed_url, headers=headers, impersonate="chrome110", timeout=15)
        
        items = []
        if feed_resp.status_code == 200:
            feed_data = feed_resp.json()
            items = feed_data.get("items", [])
            # DEBUG
            # with open("insta_feed_debug.json", "w") as f:
            #    json.dump(feed_data, f, indent=2)
        else:
            console.print(f"  [error]Feed API fails (status {feed_resp.status_code}). Falling back to profile media edges.[/]")
            timeline = user.get("edge_owner_to_timeline_media", {}).get("edges", [])
            # Convert profile edges to a common format if possible
            for edge in timeline:
                node = edge.get("node", {})
                items.append({
                    "caption": {"text": node.get("edge_media_to_caption", {}).get("edges", [{}])[0].get("node", {}).get("text", "")} if node.get("edge_media_to_caption", {}).get("edges") else None,
                    "taken_at": node.get("taken_at_timestamp"),
                    "location": node.get("location"),
                    "usertags": {"in": [ {"user": {"username": t.get("node", {}).get("user", {}).get("username")}} for t in node.get("edge_media_to_tagged_user", {}).get("edges", []) ]} if node.get("edge_media_to_tagged_user") else None
                })

        output.append(f"\nRecent Posts ({len(items)}):")
        
        for item in items:
            caption = item.get("caption", {}).get("text", "") if item.get("caption") else ""
            timestamp = item.get("taken_at")
            posted_at = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
            
            # Location
            location = item.get("location")
            location_str = f" @ {location.get('name')}" if (location and location.get('name')) else ""
            
            # Tags
            tagged_users = []
            usertags = item.get("usertags", {}).get("in", [])
            for ut in usertags:
                u_name = ut.get("user", {}).get("username")
                if u_name: tagged_users.append(f"@{u_name}")
            tags_str = f" [Tagged: {', '.join(tagged_users)}]" if tagged_users else ""

            # Stats
            likes = item.get("like_count", 0)
            views = item.get("view_count", 0)
            stats_str = f" [Likes: {likes}"
            if views: stats_str += f", Views: {views}"
            stats_str += "]"

            output.append(f"[{posted_at}{location_str}]{stats_str} {caption}{tags_str}")
            output.append("-" * 15)

        final_text = "\n".join(output)

        # Save to requested folder
        try:
            os.makedirs(folder, exist_ok=True)
            safe_name = re.sub(r'[^a-zA-Z0-9]', '_', target_username)
            prefix = "insta_" if folder == SCRAPED_DATA_FOLDER else ""
            save_path = os.path.join(folder, f"{prefix}{safe_name}.txt")
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(final_text)
            console.print(f"  [system]Instagram Saved:[/] Full data stored at {save_path}")
        except Exception as e:
            console.print(f"  [error]Failed to save Instagram data: {e}[/]")

        return final_text, "instagram.com"

    except Exception as e:
        console.print(f"  [error]Instagram Scraping failed: {e}[/]")
        return f"Failed to extract Instagram content for @{target_username}.", "instagram.com"
def get_x_cookies() -> dict:
    """Read x.com_cookies.txt and return a dictionary for curl_cffi."""
    cookies = {}
    path = "x.com_cookies.txt"
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split("\t")
                if len(parts) >= 7:
                    name = parts[5]
                    val = parts[6]
                    if val.startswith('"') and val.endswith('"'):
                        val = val[1:-1]
                    cookies[name] = val
    except Exception as e:
        console.print(f"  [error]Failed to parse {path}: {e}[/]")
    return cookies


def scrape_x(url: str, folder: str = None) -> tuple[str, str]:
    """Specialized scraper for X.com using cookies and __INITIAL_STATE__ parsing."""
    if folder is None:
        folder = X_SCRAPED_FOLDER
    cookies = get_x_cookies()
    if not cookies:
        console.print(f"  [error]No cookies found in x.com_cookies.txt. X.com scraping will likely fail.[/]")
    # Comprehensive headers for X.com
    # Using the current Bearer token discovered from main JS
    bearer_token = "Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA"
    
    # Standard headers for a desktop browser
    headers = {
        "authorization": bearer_token,
        "referer": "https://x.com/",
        "x-twitter-active-user": "yes",
        "x-twitter-client-language": "en",
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.9",
        "content-type": "application/json",
        "x-twitter-auth-type": "OAuth2Session",
    }

    target_name = url.strip("/").split("/")[-1].split("?")[0].lower()
    output = [f"Data for X User: {target_name}\n"]

    def extract_tweets_recursive(obj, found_list):
        if isinstance(obj, dict):
            # Check if this dict looks like a tweet result
            res = obj.get('result') or obj.get('tweet')
            if isinstance(res, dict):
                leg = res.get('legacy')
                if isinstance(leg, dict) and 'full_text' in leg:
                    tweet_id = res.get('rest_id') or leg.get('id_str')
                    if tweet_id not in [t['id'] for t in found_list]:
                        # 1. Start with basic full_text
                        full_text = leg.get('full_text')
                        
                        # 2. Check for NoteTweet (longform) at same level as legacy
                        note = res.get('note_tweet', {}).get('note_tweet_results', {}).get('result', {})
                        if note and note.get('text'):
                            full_text = note.get('text')
                        
                        # 3. Handle Retweets (retweeted_status_result is INSIDE legacy)
                        rt_res = leg.get('retweeted_status_result', {}).get('result', {})
                        if rt_res:
                            rt_leg = rt_res.get('legacy', {})
                            rt_text = rt_leg.get('full_text')
                            # Check if the retweeted content itself is longform
                            rt_note = rt_res.get('note_tweet', {}).get('note_tweet_results', {}).get('result', {})
                            if rt_note and rt_note.get('text'):
                                rt_text = rt_note.get('text')
                            
                            # Try to find the original author's screen name safely
                            rt_core = rt_res.get('core', {}).get('user_results', {}).get('result', {})
                            rt_author = rt_core.get('core', {}).get('screen_name') or rt_core.get('legacy', {}).get('screen_name') or "unknown"
                            
                            if rt_text:
                                full_text = f"RT @{rt_author}: {rt_text}"

                        found_list.append({
                            'id': tweet_id,
                            'text': full_text,
                            'date': leg.get('created_at'),
                            'is_reply': bool(leg.get('in_reply_to_screen_name'))
                        })
            for v in obj.values():
                extract_tweets_recursive(v, found_list)
        elif isinstance(obj, list):
            for item in obj:
                extract_tweets_recursive(item, found_list)

    try:
        session = cffi_requests.Session(impersonate="chrome110")
        session.headers.update(headers)
        session.cookies.update(cookies)

        # 2. CSRF SYNC
        home_resp = session.get("https://x.com/", timeout=15)
        fresh_csrf = session.cookies.get("ct0")
        if fresh_csrf:
             session.headers["x-csrf-token"] = fresh_csrf
             console.print(f"  [system]X.com:[/] Fresh CSRF synced: {fresh_csrf[:10]}...")
        else:
             session.headers["x-csrf-token"] = cookies.get("ct0", "")

        # 3. PROFILE FETCH (GraphQL)
        gql_profile_url = f"https://x.com/i/api/graphql/pLsOiyHJ1eFwPJlNmLp4Bg/UserByScreenName"
        profile_vars = {"screen_name": target_name, "withSafetyModeUserFields": True}
        profile_features = {"hidden_profile_subscriptions_enabled":True,"rweb_tipjar_strings_enabled":True,"responsive_web_graphql_exclude_directive_enabled":True,"verified_phone_label_enabled":False,"subscriptions_verification_info_is_identity_verified_enabled":True,"responsive_web_graphql_skip_user_profile_image_extensions_enabled":False,"responsive_web_graphql_timeline_navigation_enabled":True}
        
        profile_resp = session.get(
            gql_profile_url,
            params={"variables": json.dumps(profile_vars), "features": json.dumps(profile_features)}
        )

        user_id = ""
        if profile_resp.status_code == 200:
            user_res = profile_resp.json().get('data', {}).get('user', {}).get('result', {})
            if user_res.get('__typename') == 'User':
                legacy = user_res.get('legacy', {})
                core = user_res.get('core', {})
                user_id = user_res.get('rest_id')
                
                name = core.get('name') or legacy.get('name')
                screen_name = core.get('screen_name') or legacy.get('screen_name')
                
                output.append(f"Profile: {name or 'N/A'} (@{screen_name or target_name})")
                output.append(f"Bio: {legacy.get('description', '') or user_res.get('profile_bio', {}).get('description', '')}")
                output.append(f"Followers: {legacy.get('followers_count', 0)}")
                
                loc_data = user_res.get('location', {})
                if isinstance(loc_data, dict):
                    loc_val = loc_data.get('location', '') or legacy.get('location', '')
                else:
                    loc_val = legacy.get('location', '')
                output.append(f"Location: {loc_val}")
                
                output.append("-" * 20)
                console.print(f"  [system]X.com:[/] Profile extracted for {name} (@{screen_name or target_name})")
            else:
                console.print(f"  [error]X.com: Profile result not a 'User' (found {user_res.get('__typename')})")
        else:
             console.print(f"  [error]X.com GQL Profile failed ({profile_resp.status_code})")

        # 4. TWEETS & REPLIES FETCH
        all_tweets = []
        if user_id:
             gql_tweets_url = "https://x.com/i/api/graphql/Y59DTUMfcKmUAATiT2SlTw/UserTweets"
             tweet_vars = {"userId": user_id, "count": 20, "includePromotedContent": True, "withQuickPromoteEligibilityTweetFields": True, "withVoice": True, "withV2Timeline": True}
             tweet_features = {"rweb_tipjar_strings_enabled":True,"responsive_web_graphql_exclude_directive_enabled":True,"verified_phone_label_enabled":False,"creator_subscriptions_tweet_preview_api_enabled":True,"responsive_web_graphql_timeline_navigation_enabled":True,"responsive_web_graphql_skip_user_profile_image_extensions_enabled":False,"communities_web_enable_tweet_community_results_fetch_enabled":True,"c9s_tweet_anatomy_moderator_badge_enabled":True,"articles_preview_enabled":True,"responsive_web_edit_tweet_api_enabled":True,"graphql_is_translatable_rweb_tweet_is_translatable_enabled":True,"view_counts_everywhere_api_enabled":True,"longform_notetweets_consumption_enabled":True,"responsive_web_twitter_article_tweet_consumption_enabled":True,"tweet_awards_web_tipping_enabled":False,"creator_subscriptions_quote_tweet_preview_enabled":False,"freedom_of_speech_not_reach_fetch_enabled":True,"standardized_nudges_misinfo":True,"tweet_with_visibility_results_prefer_gql_tweet_with_visibility_results_enabled":True,"rweb_video_timestamps_enabled":True,"notetweets_Verification_info_enabled":True,"responsive_web_enhance_cards_enabled":False}
             
             tweet_resp = session.get(
                 gql_tweets_url,
                 params={"variables": json.dumps(tweet_vars), "features": json.dumps(tweet_features)}
             )

             if tweet_resp.status_code == 200:
                  extract_tweets_recursive(tweet_resp.json(), all_tweets)

        # Search Fallback (important for finding replies and mentions)
        if len(all_tweets) < 5:
             console.print(f"  [system]X.com:[/] Few tweets found. Trying SearchTimeline fallback for @{target_name}...")
             search_url = "https://x.com/i/api/graphql/NA5G_9S_L9vV_S_HTo_qBe9Ito1_A/SearchTimeline"
             search_vars = {"rawQuery": f"from:{target_name}", "count": 20, "querySource": "typed_query", "product": "Latest"}
             search_resp = session.get(search_url, params={"variables": json.dumps(search_vars), "features": json.dumps(profile_features)})
             if search_resp.status_code == 200:
                  extract_tweets_recursive(search_resp.json(), all_tweets)

        if all_tweets:
             output.append("\nRecent Posts & Comments:")
             # Sort by date if possible (Twitter dates are weird strings, simplified approach here)
             for t in all_tweets[:40]: # Show up to 40 items
                  prefix = "[REPLY]" if t['is_reply'] else "[POST]"
                  output.append(f"{prefix} [{t['date']}] {t['text']}")
                  output.append("-" * 15)
             console.print(f"  [system]X.com:[/] Found {len(all_tweets)} posts/comments for @{target_name}")

        # 5. MOBILE FALLBACK
        if len(output) <= 1:
             console.print(f"  [system]X.com:[/] GQL failed. Trying mobile-UA HTML fallback...")
             mobile_headers = headers.copy()
             mobile_headers["user-agent"] = "Mozilla/5.0 (iPhone; CPU iPhone OS 16_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Mobile/15E148 Safari/604.1"
             resp = session.get(url, headers=mobile_headers)
             
             if "window.__INITIAL_STATE__=" in resp.text:
                  console.print(f"  [system]X.com:[/] Found __INITIAL_STATE__ in mobile HTML.")
                  s_marker = "window.__INITIAL_STATE__="
                  s_idx = resp.text.find(s_marker) + len(s_marker)
                  e_idx = resp.text.find("};", s_idx) + 1
                  if s_idx > 0 and e_idx > s_idx:
                       try:
                            state = json.loads(resp.text[s_idx:e_idx])
                            entities = state.get('entities', {})
                            for uid, udata in entities.get('users', {}).get('entities', {}).items():
                                 if udata.get('screen_name', '').lower() == target_name:
                                      output.append(f"Profile: {udata.get('name')} (@{udata.get('screen_name')})")
                                      output.append(f"Bio: {udata.get('description')}")
                                      output.append("-" * 20)
                            
                            for tid, tdata in entities.get('tweets', {}).get('entities', {}).items():
                                 output.append(f"Tweet [{tdata.get('created_at')}]: {tdata.get('full_text')}")
                       except: pass

             if len(output) <= 1:
                  soup = BeautifulSoup(resp.text, "html.parser")
                  return soup.get_text(separator="\n"), "x.com"

        final_text = "\n".join(output)
        
        # Save to requested folder
        try:
            os.makedirs(folder, exist_ok=True)
            safe_name = re.sub(r'[^a-zA-Z0-9]', '_', target_name)
            prefix = "x_" if folder == SCRAPED_DATA_FOLDER else ""
            save_path = os.path.join(folder, f"{prefix}{safe_name}.txt")
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(final_text)
            console.print(f"  [system]X.com Saved:[/] Full data stored at {save_path}")
        except Exception as e:
            console.print(f"  [error]Failed to save X data: {e}[/]")

        return final_text, "x.com"

    except Exception as e:
        console.print(f"  [error]X.com Scraping failed: {e}[/]")
        return "Failed to extract X.com content.", "x.com"


def scrape_url(url: str, save_standalone: bool = False) -> tuple[str, str, list[str]]:
    """
    Fetch a URL using curl_cffi impersonating browser (or safari for reddit), 
    strip boilerplate, return (clean_text, label, downloaded_docs).
    If save_standalone is True, saves raw content to SCRAPED_DATA_FOLDER.
    """
    parsed = urllib.parse.urlparse(url)
    is_reddit = "reddit.com" in parsed.netloc
    is_x = "x.com" in parsed.netloc or "twitter.com" in parsed.netloc
    is_insta = "instagram.com" in parsed.netloc
    impersonate_target = "chrome"
    
    if is_x:
        # If standalone, use scrapped-data, else use x-scraped
        target_folder = SCRAPED_DATA_FOLDER if save_standalone else X_SCRAPED_FOLDER
        text, label = scrape_x(url, folder=target_folder)
        return text, label, []

    if is_insta:
        # If standalone, use scrapped-data, else use insta-scraped
        target_folder = SCRAPED_DATA_FOLDER if save_standalone else INSTA_SCRAPED_FOLDER
        text, label = scrape_insta(url, folder=target_folder)
        return text, label, []

    if is_reddit:
        impersonate_target = "safari15_5"
        if not parsed.path or parsed.path == "/":
            url = urllib.parse.urljoin(url, "/.json")
        else:
            url = url.rstrip("/") + ".json"
        console.print(f"  [system]Reddit Optimization:[/] Switching to .json endpoint for full thread history.")

    # Use curl_cffi to perfectly impersonate browser TLS fingerprints
    resp = cffi_requests.get(url, impersonate=impersonate_target, timeout=15)
    resp.raise_for_status()

    # Handle Reddit JSON specifically
    if is_reddit and (url.endswith(".json") or "application/json" in resp.headers.get("Content-Type", "")):
        try:
            os.makedirs(REDDIT_JSON_FOLDER, exist_ok=True)
            name_part = parsed.path.strip("/").replace("/", "_") or "homepage"
            json_path = os.path.join(REDDIT_JSON_FOLDER, f"{name_part}.json")
            
            with open(json_path, "w", encoding="utf-8") as f:
                f.write(resp.text)
            
            label = parsed.netloc or "reddit.com"
            console.print(f"  [system]Reddit Saved:[/] Raw JSON stored at {json_path}")
            
            data = json.loads(resp.text)
            text = parse_reddit_json(data)
            
            if save_standalone:
                os.makedirs(SCRAPED_DATA_FOLDER, exist_ok=True)
                txt_path = os.path.join(SCRAPED_DATA_FOLDER, f"reddit_{name_part}.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)
                console.print(f"  [system]Persisted Reddit content to:[/] {txt_path}")

            return text, label, [json_path]
        except Exception as e:
            console.print(f"  [error]Failed to parse Reddit JSON: {e}. Falling back to HTML scraping.[/]")

    # Standard HTML scraping
    soup = BeautifulSoup(resp.text, "html.parser")

    # Process links for documents
    downloaded_docs = []
    supported_exts = {".pdf", ".docx", ".txt", ".csv", ".md"}
    os.makedirs(SCRAPED_DOCS_FOLDER, exist_ok=True)
    
    for a in soup.find_all("a", href=True):
        href = a["href"]
        parsed_href = urllib.parse.urlparse(href)
        ext = os.path.splitext(parsed_href.path)[1].lower()
        if ext in supported_exts:
            abs_url = urllib.parse.urljoin(url, href)
            filename = os.path.basename(parsed_href.path)
            if not filename: continue
            save_path = os.path.join(SCRAPED_DOCS_FOLDER, filename)
            if os.path.exists(save_path):
                if save_path not in downloaded_docs: downloaded_docs.append(save_path)
                continue
            try:
                console.print(f"  [system]Downloading:[/] {abs_url}")
                doc_resp = cffi_requests.get(abs_url, impersonate="chrome", timeout=15)
                doc_resp.raise_for_status()
                with open(save_path, "wb") as f:
                    f.write(doc_resp.content)
                downloaded_docs.append(save_path)
            except Exception as e:
                console.print(f"  [error]Failed to download '{abs_url}': {e}[/]")

    # Standard HTML Cleanup
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript", "iframe"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    label = parsed.netloc or url

    if save_standalone:
        os.makedirs(SCRAPED_DATA_FOLDER, exist_ok=True)
        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', label[:50])
        txt_path = os.path.join(SCRAPED_DATA_FOLDER, f"web_{safe_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        console.print(f"  [system]Persisted web content to:[/] {txt_path}")
    else:
        # Also persist standard web content for /add-url if requested?
        # The user's prompt says "add-url then their respective folders ... scraped-docs"
        # Since standard HTML is often just indexed, but user mentions "scraped-docs"
        # Let's ensure it's saved to scraped-docs/ too if /add-url
        os.makedirs(SCRAPED_DOCS_FOLDER, exist_ok=True)
        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', label[:50])
        txt_path = os.path.join(SCRAPED_DOCS_FOLDER, f"{safe_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        console.print(f"  [system]Persisted web content to:[/] {txt_path}")

    return text, label, downloaded_docs


def add_url_to_db(
    collection: chromadb.Collection,
    url: str,
    doc_chunk_counts: dict[str, int],
    chunk_offset: int,
) -> int:
    """Scrape a URL, fetch linked documents, and embed their contents. Returns updated chunk offset."""
    console.print(f"  [system]Fetching:[/] {url}")
    text, label, downloaded_docs = scrape_url(url)
    
    # Only index in-memory text if it exists (standard HTML)
    if text.strip():
        chunks = chunk_text(text, label)
        console.print(f"  [system]Embedding {len(chunks)} chunks from '{label}' ({len(text)} chars)…[/]")
        ids, embeddings, documents, metadatas = [], [], [], []
        for i, chunk in enumerate(chunks):
            cid = f"chunk_{chunk_offset + i}"
            emb = get_embedding(chunk["text"])
            ids.append(cid)
            embeddings.append(emb)
            documents.append(chunk["text"])
            metadatas.append({"source": label})

        collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        doc_chunk_counts[label] = len(chunks)
        console.print(f"  [system]✓ '{label}' indexed — {len(chunks)} chunks.[/]")
        chunk_offset += len(chunks)
    
    # Process downloaded documents (PDFs, DOCX, and Reddit JSON)
    for doc_path in downloaded_docs:
        name = os.path.basename(doc_path)
        try:
            # Special Handling for Reddit JSON: Parse and index specifically from the FILE
            if doc_path.endswith(".json") and REDDIT_JSON_FOLDER in doc_path:
                with open(doc_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                doc_text = parse_reddit_json(raw_data)
            else:
                doc_text = load_file(doc_path)
            
            doc_chunks = chunk_text(doc_text, name)
            
            if not doc_chunks:
                continue
                
            console.print(f"  [system]Embedding {len(doc_chunks)} chunks from file '{name}' ({len(doc_text)} chars)…[/]")
            doc_ids, doc_embeddings, doc_documents, doc_metadatas = [], [], [], []
            for i, chunk in enumerate(doc_chunks):
                cid = f"chunk_{chunk_offset + i}"
                emb = get_embedding(chunk["text"])
                doc_ids.append(cid)
                doc_embeddings.append(emb)
                doc_documents.append(chunk["text"])
                doc_metadatas.append({"source": name})
                
            collection.add(ids=doc_ids, embeddings=doc_embeddings, documents=doc_documents, metadatas=doc_metadatas)
            doc_chunk_counts[name] = len(doc_chunks)
            console.print(f"  [system]✓ '{name}' indexed — {len(doc_chunks)} chunks.[/]")
            chunk_offset += len(doc_chunks)
        except Exception as e:
            console.print(f"  [error]Failed to process downloaded doc '{name}': {e}[/]")

    return chunk_offset
