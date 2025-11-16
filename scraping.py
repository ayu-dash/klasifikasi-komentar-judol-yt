import os
import time
import random
import ssl
import urllib3
from tqdm import tqdm
from googleapiclient.discovery import build
from langdetect import detect, LangDetectException
import pandas as pd

# =============================
# LIST API KEYS
# =============================
API_KEYS = [
    "API_KEY"
]

current_key_index = 0

def get_youtube_service():
    global current_key_index
    key = API_KEYS[current_key_index]
    return build("youtube", "v3", developerKey=key)

youtube = get_youtube_service()

def switch_api_key():
    """Ganti API key ke berikutnya kalau kuota habis."""
    global current_key_index, youtube
    current_key_index = (current_key_index + 1) % len(API_KEYS)
    youtube = get_youtube_service()
    print(f"🔄 Ganti ke API key #{current_key_index + 1}")

# =============================
# KONFIGURASI
# =============================
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

keywords = ["slot", "toto", "casino", "poker", "rtp", "jackpot", "gacor", "zeus"]
custom_video_ids = ["YZ4N8jH5R_M", "s9OU_mLo-KU", "1msXOdJcG9s", "Nkh1KiTS5CM", 
                   "rkoymgMW-8M", "7TsgXbRGOQo", "yY76VsIplzo", "JpaK8OhL4FI", 
                   "UnVihN2_M2U", "GHbSjBdMB8E", "4k6rzuj0bWI", "FpSJFqYaRb8", "dXtcUtRJO0g"]

max_videos_per_keyword = 100
max_comments_per_video = 500

# =============================
# FUNGSI BANTU
# =============================
def is_indonesian(text):
    """Coba deteksi bahasa; fallback ke True jika pendek/ambigu."""
    text = text.strip()
    if len(text) < 5:  # komentar sangat pendek (emoji, dll)
        return True
    try:
        return detect(text) == "id"
    except LangDetectException:
        return True  # biar gak di-drop diam-diam

def is_promo_comment(text):
    promo_words = [
        "toto", "slot", "gacor", "maxwin", "casino", "angka hoki",
        "bonus", "deposit", "jackpot", "akun", "daftar", "situs",
        "wd", "link", "klik", "spin", "rtp", "bet", "bo", "scatter"
    ]
    t = text.lower()
    return any(word in t for word in promo_words)

def search_videos(keyword):
    vids = []
    next_page_token = None
    while len(vids) < max_videos_per_keyword:
        try:
            req = youtube.search().list(
                q=keyword,
                part="id",
                type="video",
                maxResults=50,
                pageToken=next_page_token,
                regionCode="ID"
            )
            res = req.execute()
            for item in res.get("items", []):
                vids.append(item["id"]["videoId"])
                if len(vids) >= max_videos_per_keyword:
                    break
            next_page_token = res.get("nextPageToken")
            if not next_page_token:
                break
        except Exception as e:
            msg = str(e)
            if "quotaExceeded" in msg:
                print("⚠️ Kuota habis, ganti API key...")
                switch_api_key()
                continue
            print(f"⚠️ Error cari video: {msg}")
            break
        time.sleep(random.uniform(0.5, 1.2))
    return vids

def get_comments(video_id):
    comments = []
    skipped_reason = None
    next_token = None
    page_counter = 0

    with tqdm(total=max_comments_per_video, desc=f"💬 {video_id}", leave=False) as pbar:
        while len(comments) < max_comments_per_video:
            try:
                req = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    textFormat="plainText",
                    maxResults=100,
                    pageToken=next_token
                )
                res = req.execute()
                items = res.get("items", [])
                page_counter += 1

                if not items:
                    skipped_reason = "no_items"
                    break

                for item in items:
                    snippet = item["snippet"]["topLevelComment"]["snippet"]
                    text = snippet["textOriginal"].strip()
                    if is_indonesian(text):
                        comments.append({
                            "video_id": video_id,
                            "author": snippet["authorDisplayName"],
                            "comment_text": text,
                            "published_at": snippet["publishedAt"],
                            "like_count": snippet.get("likeCount", 0),
                            "is_promo": is_promo_comment(text)
                        })
                        pbar.update(1)
                        if len(comments) >= max_comments_per_video:
                            break

                next_token = res.get("nextPageToken")
                if not next_token:
                    break

            except Exception as e:
                msg = str(e)
                if "commentsDisabled" in msg:
                    skipped_reason = "comments_disabled"
                    print(f"🚫 Komentar dimatikan: {video_id}")
                    break
                elif "forbidden" in msg or "403" in msg:
                    skipped_reason = "forbidden"
                    print(f"🚫 Akses dilarang (403): {video_id}")
                    break
                elif "quotaExceeded" in msg:
                    print(f"⚠️ Kuota habis di key #{current_key_index + 1}, ganti...")
                    switch_api_key()
                    continue
                else:
                    skipped_reason = msg
                    print(f"⚠️ Error ambil komentar {video_id}: {msg}")
                    time.sleep(1)
                    continue
            time.sleep(random.uniform(0.6, 1.2))

    # Catat kalau tidak ada hasil sama sekali
    if len(comments) == 0:
        reason = skipped_reason or "unknown_empty"
        skipped_log.append({"video_id": video_id, "reason": reason, "pages_tried": page_counter})

    return comments

# =============================
# MAIN LOOP
# =============================
all_comments = []
skipped_log = []

# Ambil komentar dari video yang dipilih sendiri
print("🎯 Mengambil komentar dari video yang ditentukan...")
for i, vid in enumerate(custom_video_ids, start=1):
    print(f"🧩 Ambil komentar dari video {i}/{len(custom_video_ids)}: {vid}")
    cmts = get_comments(vid)
    if cmts:
        all_comments.extend(cmts)
        print(f"✅ Dapat {len(cmts)} komentar dari {vid}")
    else:
        print(f"⚠️ Tidak ada komentar dari video {vid}")

# Ambil komentar dari pencarian keyword (opsional)
print("\n🎯 Mencari video dengan kata kunci...")
for kw in keywords:
    print(f"\n🔍 Mencari video dengan kata kunci: '{kw}'")
    vids = search_videos(kw)
    print(f"📹 Ditemukan {len(vids)} video untuk keyword '{kw}'")
    
    for i, vid in enumerate(vids, start=1):
        # Skip jika video ID sudah diambil sebelumnya
        if any(comment['video_id'] == vid for comment in all_comments):
            print(f"⏭️ Video {vid} sudah diambil, skip...")
            continue
            
        print(f"🧩 Ambil komentar dari video {i}/{len(vids)}: {vid}")
        cmts = get_comments(vid)
        if cmts:
            all_comments.extend(cmts)
            print(f"✅ Dapat {len(cmts)} komentar dari {vid}")
        else:
            print(f"⚠️ Tidak ada komentar dari video {vid}")

# =============================
# SIMPAN HASIL
# =============================
df = pd.DataFrame(all_comments)
df.to_csv("comments_from_scraping.csv", index=False, encoding="utf-8-sig")

if skipped_log:
    pd.DataFrame(skipped_log).to_csv("skipped_videos.csv", index=False, encoding="utf-8-sig")
    print(f"\n⚠️ {len(skipped_log)} video gagal diambil, disimpan ke skipped_videos.csv")

print(f"\n✅ Total komentar terkumpul: {len(all_comments)}")
print("💾 Disimpan ke comments_from_scraping.csv")

# Tampilkan statistik
if len(all_comments) > 0:
    promo_count = sum(1 for comment in all_comments if comment['is_promo'])
    print(f"📊 Statistik:")
    print(f"   - Total komentar: {len(all_comments)}")
    print(f"   - Komentar promosi: {promo_count} ({promo_count/len(all_comments)*100:.1f}%)")
    print(f"   - Video unik: {len(set(comment['video_id'] for comment in all_comments))}")
