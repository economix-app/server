import os
import json
import time
import random
import logging
from uuid import uuid4
from threading import Thread
from typing import Dict, Optional, Tuple

from flask import Flask, request, jsonify, send_file, redirect
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from hashlib import sha256
from functools import wraps
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError
import re
import html
import pyotp
import qrcode
import io
from better_profanity import profanity
import requests
from logging.handlers import RotatingFileHandler

# Constants
ITEM_CREATE_COOLDOWN = 60  # 1 minute
TOKEN_MINE_COOLDOWN = 300  # 5 minutes
MAX_ITEM_PRICE = 1000000000000
MIN_ITEM_PRICE = 1

# Application Setup
app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get("FLASK_SECRET_KEY", "1234"),
)
CORS(app, origins=os.environ.get("CORS_ORIGINS", "").split(","))

# Logging Configuration
handler = RotatingFileHandler("app.log", maxBytes=10*1024*1024, backupCount=5)
handler.setFormatter(logging.Formatter(
    "%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]"
))
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# Database Setup
client = MongoClient(os.environ.get("MONGODB_URI"), maxPoolSize=50, connect=False)
db = client[os.environ.get("MONGODB_DB")]
Collections = {
    'users': db.users,
    'items': db.items,
    'messages': db.messages,
    'item_meta': db.item_meta,
    'misc': db.misc,
    'pets': db.pets,
    'account_creation_attempts': db.account_creation_attempts,
    'message_attempts': db.message_attempts,
    'blocked_ips': db.blocked_ips,
    'failed_logins': db.failed_logins,
    'user_history': db.user_history
}

# AutoMod Configuration
AUTOMOD_CONFIG = {
    "ACCOUNT_CREATION_THRESHOLD": 7,
    "ACCOUNT_CREATION_TIME_WINDOW": 60,
    "MESSAGE_SPAM_THRESHOLD": 5,
    "MESSAGE_SPAM_TIME_WINDOW": 3,
    "MESSAGE_SPAM_MUTE_DURATION": "5m",
    "NEW_USER_MESSAGE_SPAM_MUTE_DURATION": "10m",
    "ACCOUNT_CREATION_BLOCK_DURATION": 300,
    "MESSAGE_IP_THRESHOLD": 15,
    "MESSAGE_IP_WINDOW": 5,
    "FAILED_LOGIN_THRESHOLD": 5,
    "FAILED_LOGIN_WINDOW": 60,
    "MAX_LINKS": 2,
    "MAX_CAPS_RATIO": 0.7,
    "MIN_ACCOUNT_AGE": 3600,
    "SUBNET_BLOCKING": True,
    "SPAM_PATTERNS": [
        r"(?i)free\s+money",
        r"(?i)buy\s+followers",
        r"(?i)http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        r"(?i)cheap\s+loans",
        r"(?i)work\s+from\s+home",
        r"(?i)make\s+money\s+fast",
        r"(?i)weight\s+loss\s+supplements",
        r"(?i)earn\s+\$\d+\s+per\s+day",
        r"(?i)limited\s+time\s+offer",
        r"(?i)click\s+here\s+to\s+claim",
        r"(?i)congratulations\s+you\s+won",
        r"(?i)instant\s+cash",
        r"(?i)no\s+credit\s+check\s+loans",
        r"(?i)100%\s+guaranteed",
        r"(?i)miracle\s+cure",
        r"(?i)risk\s+free\s+trial",
        r"(?i)as\s+seen\s+on\s+TV",
        r"(?i)win\s+a\s+free\s+(iPhone|gift card|vacation)",
        r"(?i)boost\s+your\s+SEO\s+ranking",
        r"(?i)hot\s+singles\s+in\s+your\s+area",
        r"(?i)order\s+now\s+and\s+save",
        r"(?i)act\s+now\s+before\s+it's\s+gone",
    ]
}

# Load Word Lists
def load_word_lists():
    global ADJECTIVES, MATERIALS, NOUNS, SUFFIXES, PET_NAMES
    try:
        with open("words/adjectives.json") as f: ADJECTIVES = json.load(f)
        with open("words/materials.json") as f: MATERIALS = json.load(f)
        with open("words/nouns.json") as f: NOUNS = json.load(f)
        with open("words/suffixes.json") as f: SUFFIXES = json.load(f)
        with open("words/pet_names.json") as f: PET_NAMES = json.load(f)
        app.logger.info("Loaded item generation word lists successfully")
    except Exception as e:
        app.logger.critical(f"Failed to load word lists: {str(e)}")
        raise

load_word_lists()
profanity.load_censor_words()

# Index Creation
def create_indexes():
    Collections['users'].create_index([("username", ASCENDING)], unique=True)
    Collections['items'].create_index([("id", ASCENDING), ("owner", ASCENDING)])
    Collections['messages'].create_index([("room", ASCENDING), ("timestamp", ASCENDING)])
    Collections['item_meta'].create_index([("id", ASCENDING)])
    Collections['misc'].create_index([("type", ASCENDING)])
    Collections['pets'].create_index([("id", ASCENDING)], unique=True)
    Collections['account_creation_attempts'].create_index(
        [("timestamp", ASCENDING)], 
        expireAfterSeconds=AUTOMOD_CONFIG["ACCOUNT_CREATION_TIME_WINDOW"]
    )
    Collections['message_attempts'].create_index(
        [("timestamp", ASCENDING)], 
        expireAfterSeconds=AUTOMOD_CONFIG["MESSAGE_SPAM_TIME_WINDOW"]
    )
    Collections['blocked_ips'].create_index([("blocked_until", ASCENDING)], expireAfterSeconds=0)
    Collections['blocked_ips'].create_index([("ip", ASCENDING)])
    Collections['failed_logins'].create_index(
        [("timestamp", ASCENDING)], 
        expireAfterSeconds=AUTOMOD_CONFIG["FAILED_LOGIN_WINDOW"]
    )
    Collections['message_attempts'].create_index([("ip", ASCENDING), ("timestamp", ASCENDING)])
    Collections['user_history'].create_index([("username", ASCENDING)])

create_indexes()

# Utility Functions
def split_name(name: str) -> Dict[str, str]:
    parts = name.split(" ")
    return {
        "adjective": parts[0],
        "material": parts[1],
        "noun": parts[2],
        "suffix": " ".join(parts[3:]).split("#")[0],
        "number": " ".join(parts[3:]).split("#")[1],
    }

def get_level(rarity: float) -> str:
    thresholds = [
        (0.1, "Godlike"), (1, "Legendary"), (5, "Epic"),
        (10, "Rare"), (25, "Uncommon"), (50, "Common"),
        (75, "Scrap")
    ]
    for threshold, level in thresholds:
        if rarity <= threshold:
            return level
    return "Trash"

def parse_time(length: str) -> int:
    if not length or length.lower() == "perma":
        return 0
    duration = 0
    for part in length.split("+"):
        value = int(part[:-1])
        unit = part[-1].lower()
        multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800, "y": 31536000}
        duration += value * multipliers.get(unit, 0)
    return int(time.time()) + duration

def check_content_spam(message: str) -> bool:
    config = AUTOMOD_CONFIG
    for pattern in config["SPAM_PATTERNS"]:
        if re.search(pattern, message):
            return True
    if len(re.findall(r"http[s]?://", message)) > config["MAX_LINKS"]:
        return True
    if len(message) > 10:
        caps_count = sum(1 for c in message if c.isupper())
        if caps_count / len(message) > config["MAX_CAPS_RATIO"]:
            return True
    return False

def send_discord_notification(title: str, description: str, color: int = 0x00FF00):
    webhook_url = os.environ.get("DISCORD_WEBHOOK")
    if not webhook_url:
        app.logger.error("Discord webhook URL not configured")
        return
    
    def _send():
        data = {"embeds": [{"title": title, "description": description, "color": color}]}
        response = requests.post(webhook_url, json=data)
        if response.status_code != 204:
            app.logger.error(f"Discord notification failed: {response.status_code}")
    
    Thread(target=_send).start()

# Authentication Decorators
def requires_admin(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        user = Collections['users'].find_one({"username": request.username})
        if user.get("type") != "admin":
            return jsonify({"error": "Admin privileges required", "code": "admin-required"}), 403
        return f(*args, **kwargs)
    return decorated

def requires_mod(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        user = Collections['users'].find_one({"username": request.username})
        if user.get("type") not in ["admin", "mod"]:
            return jsonify({"error": "Mod privileges required", "code": "mod-required"}), 403
        return f(*args, **kwargs)
    return decorated

def requires_unbanned(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        user = Collections['users'].find_one({"username": request.username})
        if user.get("banned_until") and (user["banned_until"] > time.time() or user["banned_until"] == 0):
            return jsonify({"error": "You are banned", "code": "banned"}), 403
        return f(*args, **kwargs)
    return decorated

# Middleware
@app.before_request
def authenticate_user():
    public_endpoints = ["register_endpoint", "login_endpoint", "index", "stats_endpoint"]
    if request.method == "OPTIONS" or request.endpoint in public_endpoints:
        return
    
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        return jsonify({"error": "Missing or invalid Authorization header", "code": "invalid-credentials"}), 401
    
    token = auth.split(" ")[1]
    user = Collections['users'].find_one({"token": token})
    if not user:
        return jsonify({"error": "Invalid token", "code": "invalid-credentials"}), 401
    
    request.username = user["username"]
    request.user_type = user.get("type", "user")

# Database Updaters
def update_item(item_id: str):
    item = Collections['items'].find_one({"id": item_id})
    if not item:
        return
    
    name = item["name"]
    meta_id = item.get("meta_id") or sha256(
        f"{name['adjective']}{name['material']}{name['noun']}{name['suffix']}".encode()
    ).hexdigest()
    
    meta = Collections['item_meta'].find_one({"id": meta_id})
    if not meta:
        rarity = round(random.uniform(0.1, 100), 1)
        meta = {
            "id": meta_id,
            "adjective": name["adjective"],
            "material": name["material"],
            "noun": name["noun"],
            "suffix": name["suffix"],
            "rarity": rarity,
            "level": get_level(rarity),
            "patented": False,
            "patent_owner": None,
            "price_history": []
        }
        Collections['item_meta'].insert_one(meta)
    
    updates = {"meta_id": meta_id, "rarity": meta["rarity"], "level": meta["level"]}
    if "history" not in item:
        updates["history"] = []
    Collections['items'].update_one({"id": item_id}, {"$set": updates})

def update_pet(pet_id: str):
    pet = Collections['pets'].find_one({"id": pet_id})
    if not pet:
        return
      
    defaults = {
        "alive": True,
        "last_fed": int(time.time()),
        "level": 1,
        "exp": 0,
        "benefits": {"token_bonus": 1},
        "health": "healthy",
        "base_price": 100
    }
    updates = {}
    for key, value in defaults.items():
        if key not in pet:
            updates[key] = value
    if updates:
        Collections['pets'].update_one({"id": pet_id}, {"$set": updates})
        
    pet = Collections['pets'].find_one({"id": pet_id})

    last_fed = pet["last_fed"]
    now = int(time.time())
    days_unfed = (now - last_fed) // (24 * 3600)  # Convert seconds to days

    # Health status and death check
    if days_unfed >= 3:
        Collections['pets'].update_one(
            {"id": pet_id},
            {"$set": {"health": "dead", "alive": False}}
        )
        send_discord_notification(
            "Pet Died",
            f"User {pet['owner']}'s pet {pet['name']} died due to neglect.",
            0xFF0000
        )
    elif days_unfed == 2:
        Collections['pets'].update_one({"id": pet_id}, {"$set": {"health": "starving"}})
    elif days_unfed == 1:
        Collections['pets'].update_one({"id": pet_id}, {"$set": {"health": "hungry"}})
    else:
        Collections['pets'].update_one({"id": pet_id}, {"$set": {"health": "healthy"}})

    # Update benefits based on level (only if alive)
    if pet["alive"]:
        pet["benefits"]["token_bonus"] = pet["level"]  # +1 token per level
        Collections['pets'].update_one({"id": pet_id}, {"$set": {"benefits": pet["benefits"]}})

def level_up_pet(pet_id: str, exp_gain: int):
    pet = Collections['pets'].find_one({"id": pet_id})
    if not pet or not pet["alive"]:
        return

    new_exp = pet["exp"] + exp_gain
    next_level_exp = exp_for_level(pet["level"] + 1)
    if new_exp >= next_level_exp:
        Collections['pets'].update_one(
            {"id": pet_id},
            {"$set": {"level": pet["level"] + 1, "exp": new_exp - next_level_exp}}
        )
        send_discord_notification(
            "Pet Leveled Up",
            f"User {pet['owner']}'s pet {pet['name']} reached level {pet['level'] + 1}!",
            0x00FF00
        )
    else:
        Collections['pets'].update_one({"id": pet_id}, {"$set": {"exp": new_exp}})

def update_account(username: str) -> Optional[Tuple[dict, int]]:
    user = Collections['users'].find_one({"username": username})
    if not user:
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404
    
    defaults = {
        "banned_until": None, "banned_reason": None, "banned": False,
        "history": [], "exp": 0, "level": 1, "frozen": False,
        "muted": False, "muted_until": None, "inventory_visibility": "private",
        "2fa_enabled": False, "pets": []
    }
    updates = {k: v for k, v in defaults.items() if k not in user}
    if updates:
        Collections['users'].update_one({"username": username}, {"$set": updates})
    
    current_time = time.time()
    if user.get("banned_until") and user["banned_until"] < current_time and user["banned_until"] != 0:
        Collections['users'].update_one(
            {"username": username},
            {"$set": {"banned_until": None, "banned_reason": None, "banned": False}}
        )
    if user.get("muted_until") and user["muted_until"] < current_time and user["muted_until"] != 0:
        Collections['users'].update_one(
            {"username": username},
            {"$set": {"muted": False, "muted_until": None}}
        )
    
    for item_id in user["items"]:
        update_item(item_id)
    if len(user["pets"]) > 1:
        refund = (len(user["pets"]) - 1) * 100
        Collections['users'].update_one(
            {"username": username},
            {"$set": {"pets": [user["pets"][0]]}, "$inc": {"tokens": refund}}
        )
    for pet_id in user["pets"]:
        update_pet(pet_id)

# Item and Pet Generation
def generate_item(owner: str) -> dict:
    def weighted_choice(items: dict, special_case: bool = False):
        choices, weights = zip(*items.items())
        if special_case:
            weights = [1/items[c]["rarity"] for c in choices]
        return random.choices(choices, weights=weights, k=1)[0]
    
    noun = weighted_choice(NOUNS, special_case=True)
    name = {
        "adjective": weighted_choice(ADJECTIVES),
        "material": weighted_choice(MATERIALS),
        "noun": noun,
        "suffix": weighted_choice(SUFFIXES),
        "number": random.randint(1, 9999),
        "icon": NOUNS[noun]["icon"]
    }
    meta_id = sha256(
        f"{name['adjective']}{name['material']}{name['noun']}{name['suffix']}".encode()
    ).hexdigest()
    
    meta = Collections['item_meta'].find_one({"id": meta_id})
    if not meta:
        rarity = round(random.uniform(0.05, 100), 2)
        meta = {
            "id": meta_id, "adjective": name["adjective"], "material": name["material"],
            "noun": name["noun"], "suffix": name["suffix"], "rarity": rarity,
            "level": get_level(rarity), "patented": False, "patent_owner": None,
            "price_history": []
        }
        Collections['item_meta'].insert_one(meta)
    
    return {
        "id": str(uuid4()), "meta_id": meta_id, "item_secret": str(uuid4()),
        "rarity": meta["rarity"], "level": meta["level"], "name": name,
        "history": [], "for_sale": False, "price": 0, "owner": owner,
        "created_at": int(time.time())
    }

def generate_pet(owner: str, base_price: int = 100) -> dict:
    return {
        "id": str(uuid4()),
        "name": random.choice(PET_NAMES),
        "level": 1,
        "exp": 0,
        "owner": owner,
        "created_at": int(time.time()),
        "last_fed": int(time.time()),
        "health": "healthy",
        "benefits": {"token_bonus": 1},
        "alive": True,
        "base_price": base_price,
    }

# Experience System
def exp_for_level(level: int) -> int:
    return int(25 * (1.2 ** (level - 1)))

def add_exp(username: str, exp: int):
    user = Collections['users'].find_one({"username": username})
    if not user:
        return
    new_exp = user["exp"] + exp
    Collections['users'].update_one({"username": username}, {"$set": {"exp": new_exp}})
    if new_exp >= exp_for_level(user["level"] + 1):
        Collections['users'].update_one(
            {"username": username}, 
            {"$set": {"level": user["level"] + 1}}
        )

def set_exp(username: str, exp: int):
    user = Collections['users'].find_one({"username": username})
    if not user:
        return
    Collections['users'].update_one({"username": username}, {"$set": {"exp": exp}})
    if exp >= exp_for_level(user["level"] + 1):
        Collections['users'].update_one(
            {"username": username}, 
            {"$set": {"level": user["level"] + 1}}
        )

def set_level(username: str, level: int):
    user = Collections['users'].find_one({"username": username})
    if not user:
        return
    level_exp = exp_for_level(level)
    Collections['users'].update_one(
        {"username": username}, 
        {"$set": {"level": level, "exp": level_exp}}
    )

# Core Handlers
def register(username: str, password: str, ip: str) -> Tuple[dict, int]:
    if not username or not password:
        return jsonify({"error": "Missing credentials", "code": "missing-credentials"}), 400
    
    if Collections['blocked_ips'].find_one({"ip": ip, "blocked_until": {"$gt": time.time()}}):
        return jsonify({"error": "IP blocked", "code": "ip-blocked"}), 429
    
    current_time = time.time()
    recent_attempts = Collections['account_creation_attempts'].count_documents({
        "ip": ip,
        "timestamp": {"$gt": current_time - AUTOMOD_CONFIG["ACCOUNT_CREATION_TIME_WINDOW"]}
    })
    
    if recent_attempts >= AUTOMOD_CONFIG["ACCOUNT_CREATION_THRESHOLD"]:
        blocked_until = current_time + AUTOMOD_CONFIG["ACCOUNT_CREATION_BLOCK_DURATION"]
        Collections['blocked_ips'].update_one(
            {"ip": ip},
            {"$set": {"blocked_until": blocked_until, "timestamp": current_time}},
            upsert=True
        )
        Collections['users'].update_many(
            {"creation_ip": ip},
            {"$set": {"banned": True, "banned_until": blocked_until, "banned_reason": "Account spam"}}
        )
        Collections['messages'].insert_one({
            "id": str(uuid4()), 
            "room": "global", 
            "username": "AutoMod",
            "message": f"""
            <p><span style="color: #FF5555">[WARNING]</span> Detected <b>{recent_attempts + 1}x</b> Account Creation Spam</p>
            <p>IP: <b>{ip}</b> has been blocked for <b>{AUTOMOD_CONFIG['ACCOUNT_CREATION_BLOCK_DURATION']} seconds</b></p>
            """,
            "timestamp": current_time, 
            "type": "system"
        })
        send_discord_notification(
            "AutoMod Action",
            f"Blocked IP {ip} for account spam. Banned {recent_attempts + 1} accounts.",
            0xFF0000
        )
        return jsonify({"error": "Account spam detected", "code": "account-spam"}), 429
    
    username = profanity.censor(username.strip(), censor_char="-")
    if not re.match(r"^[a-zA-Z0-9_-]{3,20}$", username):
        return jsonify({"error": "Invalid username"}), 400
    
    try:
        user_data = {
            "created_at": int(time.time()), "username": username,
            "password_hash": generate_password_hash(password), "type": "user",
            "tokens": 100, "last_item_time": 0, "last_mine_time": 0, "items": [],
            "token": None, "banned_until": None, "banned_reason": None, "banned": False,
            "muted": False, "muted_until": None, "history": [], "exp": 0, "level": 1,
            "2fa_enabled": False, "inventory_visibility": "private", "pets": [],
            "creation_ip": ip
        }
        Collections['users'].insert_one(user_data)
        Collections['account_creation_attempts'].insert_one({"ip": ip, "timestamp": current_time})
        send_discord_notification("New user registered", f"Username: {username}")
        return jsonify({"success": True}), 201
    except DuplicateKeyError:
        return jsonify({"error": "Username exists", "code": "username-exists"}), 400

def login(username: str, password: str, ip: str, code: Optional[str] = None, 
         token: Optional[str] = None) -> Tuple[dict, int]:
    recent_fails = Collections['failed_logins'].count_documents({
        "ip": ip,
        "timestamp": {"$gt": time.time() - AUTOMOD_CONFIG["FAILED_LOGIN_WINDOW"]}
    })
    if recent_fails >= AUTOMOD_CONFIG["FAILED_LOGIN_THRESHOLD"]:
        blocked_until = time.time() + AUTOMOD_CONFIG["FAILED_LOGIN_WINDOW"]
        Collections['blocked_ips'].insert_one({
            "ip": ip, "subnet": ".".join(ip.split(".")[:3]) + ".0/24",
            "blocked_until": blocked_until, "reason": "Too many failed logins"
        })
        return jsonify({"error": "Too many failed attempts", "code": "login-locked"}), 429
    
    user = Collections['users'].find_one({"username": username})
    if not user or not check_password_hash(user["password_hash"], password):
        Collections['failed_logins'].insert_one({"ip": ip, "timestamp": time.time()})
        return jsonify({"error": "Invalid credentials", "code": "invalid-credentials"}), 401
    
    if user.get("2fa_enabled", False):
        if not code and not token:
            return jsonify({"error": "2FA required", "code": "2fa-required"}), 401
        if code:
            if user["2fa_code"] != code:
                return jsonify({"error": "Invalid 2FA code", "code": "invalid-2fa-code"}), 401
        else:
            totp = pyotp.TOTP(user["2fa_secret"])
            if not totp.verify(token):
                return jsonify({"error": "Invalid 2FA token", "code": "invalid-2fa-token"}), 401
    
    token = str(uuid4())
    Collections['users'].update_one({"username": username}, {"$set": {"token": token}})
    send_discord_notification("User logged in", f"Username: {username}")
    return jsonify({"success": True, "token": token})

def get_users() -> Tuple[dict, int]:
    users = Collections['users'].find({}, {"_id": 0, "username": 1})
    return jsonify({"usernames": [user["username"] for user in users]})

def parse_command(username: str, command: str, room_name: str) -> str:
    user = Collections['users'].find_one({"username": username})
    is_admin = user.get("type") == "admin"
    is_mod = user.get("type") in ["admin", "mod"]
    
    parts = command[1:].split(" ")
    cmd, *args = parts
    
    if cmd == "clear_chat" and is_admin:
        Collections['messages'].delete_many({"room": room_name})
        return f"Cleared chat in <b>{room_name}</b>"
    elif cmd == "clear_user" and len(args) == 1 and is_admin:
        Collections['messages'].delete_many({"room": room_name, "username": args[0]})
        return f"Deleted messages from <b>{args[0]}</b> in <b>{room_name}</b>"
    elif cmd == "delete_many" and len(args) == 1 and is_admin:
        try:
            amount = int(args[0])
            messages = Collections['messages'].find({"room": room_name}).sort("timestamp", DESCENDING).limit(amount)
            ids = [doc["_id"] for doc in messages]
            Collections['messages'].delete_many({"_id": {"$in": ids}})
            return f"Deleted <b>{amount}</b> messages from <b>{room_name}</b>"
        except ValueError:
            return "Invalid amount specified"
    elif cmd == "ban" and len(args) >= 3 and is_admin:
        target, duration, *reason = args
        ban_user(target, duration, " ".join(reason))
        return f"Banned <b>{target}</b> for <b>{' '.join(reason)}</b> (<b>{duration}</b>)"
    elif cmd == "mute" and len(args) == 2 and is_mod:
        mute_user(args[0], args[1])
        return f"Muted <b>{args[0]}</b> for <b>{args[1]}</b>"
    elif cmd == "unban" and len(args) == 1 and is_admin:
        Collections['users'].update_one({"username": args[0]}, {"$set": {"banned": False}})
        return f"Unbanned <b>{args[0]}</b>"
    elif cmd == "unmute" and len(args) == 1 and is_mod:
        unmute_user(args[0])
        return f"Unmuted <b>{args[0]}</b>"
    elif cmd == "sudo" and len(args) >= 2 and is_admin:
        sudo_user = args[0]
        message = " ".join(args[1:])
        user_data = Collections['users'].find_one({"username": sudo_user})
        if not user_data:
            return f"User <b>{sudo_user}</b> not found"
        Collections['messages'].insert_one({
            "id": str(uuid4()), "room": room_name, "username": sudo_user,
            "message": message, "timestamp": time.time(), "type": user_data["type"]
        })
        return None
    elif cmd == "list_banned" and is_mod:
        banned = Collections['users'].find({"banned": True})
        banned_list = [f"<b>{u['username']}</b> - {u.get('banned_reason', 'No reason')}" for u in banned]
        return "Banned users:\n" + "\n".join(banned_list) if banned_list else "Nobody is banned."
    elif cmd == "help" and is_mod:
        return """
        <h3>Available Commands:</h3>
        <h4>Admin</h4>
        <p>/clear_chat - Clears all messages in the current room</p>
        <p>/clear_user [username] - Clears all messages from a specific user</p>
        <p>/delete_many [amount] - Deletes specified number of messages</p>
        <p>/ban [username] [duration] [reason] - Bans a user</p>
        <p>/unban [username] - Unbans a user</p>
        <p>/sudo [username] [message] - Sends message as user</p>
        <h4>Admin & Mod</h4>
        <p>/mute [username] [duration] - Mutes a user</p>
        <p>/unmute [username] - Unmutes a user</p>
        <p>/list_banned - Lists banned users</p>
        <p>/help - Shows this help</p>
        """
    return "Invalid command"

def send_message(room_name: str, message_content: str, username: str, ip: str) -> Tuple[dict, int]:
    user = Collections['users'].find_one({"username": username})
    if user["muted"]:
        return jsonify({"error": "You are muted", "code": "user-muted"}), 400
    
    if not room_name or not message_content:
        return jsonify({"error": "Missing room or message", "code": "missing-parameters"}), 400
    
    if not re.match(r"^[a-zA-Z0-9_-]{1,50}$", room_name):
        return jsonify({"error": "Invalid room name", "code": "invalid-room"}), 400
    
    current_time = time.time()
    message_count = Collections['message_attempts'].count_documents({
        "username": username,
        "timestamp": {"$gt": current_time - AUTOMOD_CONFIG["MESSAGE_SPAM_TIME_WINDOW"]}
    })
    
    if check_content_spam(message_content):
        Collections['messages'].delete_many({"username": username, "timestamp": {"$gt": time.time() - 2}})
        Collections['user_history'].insert_one({
            "username": username, "type": "content_spam", "timestamp": time.time(),
            "details": message_content[:100]
        })
        mute_user(username, AUTOMOD_CONFIG["MESSAGE_SPAM_MUTE_DURATION"], notify=False)
        return jsonify({"error": "Message blocked", "code": "content-spam"}), 403
    
    if message_count >= AUTOMOD_CONFIG["MESSAGE_SPAM_THRESHOLD"]:
        is_new = (current_time - user["created_at"]) < AUTOMOD_CONFIG["MIN_ACCOUNT_AGE"]
        mute_duration = (AUTOMOD_CONFIG["NEW_USER_MESSAGE_SPAM_MUTE_DURATION"] if is_new 
                        else AUTOMOD_CONFIG["MESSAGE_SPAM_MUTE_DURATION"])
        mute_user(username, mute_duration, notify=False)
        deleted = Collections['messages'].delete_many({
            "username": username,
            "timestamp": {"$gt": current_time - AUTOMOD_CONFIG["MESSAGE_SPAM_TIME_WINDOW"]}
        }).deleted_count
        Collections['messages'].insert_one({
            "id": str(uuid4()), 
            "room": "global", 
            "username": "AutoMod",
            "message": f"""
            <p><span style="color: #FF5555">[WARNING]</span> Detected <b>{deleted}x</b> Message Spam</p>
            <p>User: <b>{username}</b> has been muted for <b>{mute_duration}</b></p>
            """,
            "timestamp": current_time, 
            "type": "system"
        })
        send_discord_notification(
            "AutoMod Action",
            f"Muted {username} for spamming. Deleted {deleted} messages.",
            0xFF0000
        )
        return jsonify({"error": "Message spam detected", "code": "message-spam"}), 429
    
    Collections['message_attempts'].insert_one({"username": username, "ip": ip, "timestamp": current_time})
    sanitized_message = profanity.censor(html.escape(message_content.strip()))
    if not sanitized_message:
        return jsonify({"error": "Message empty", "code": "empty-message"}), 400
    if len(sanitized_message) > 100:
        return jsonify({"error": "Message too long", "code": "message-too-long"}), 400
    
    if user["type"] in ["admin", "mod"] and sanitized_message.startswith("/"):
        system_message = parse_command(username, sanitized_message, room_name)
        if system_message:
            Collections['messages'].insert_one({
                "id": str(uuid4()), 
                "room": room_name, 
                "username": "Command Handler",
                "message": system_message, 
                "timestamp": time.time(), 
                "type": "system"
            })
    else:
        Collections['messages'].insert_one({
            "id": str(uuid4()), 
            "room": room_name, 
            "username": username,
            "message": sanitized_message, 
            "timestamp": time.time(), 
            "type": user["type"]
        })
    return jsonify({"success": True})

# Admin/Mod Functions
def reset_cooldowns(username: str) -> Tuple[dict, int]:
    Collections['users'].update_one(
        {"username": username}, {"$set": {"last_item_time": 0, "last_mine_time": 0}}
    )
    send_discord_notification(
        "Cooldowns Reset",
        f"Admin {request.username} reset cooldowns for {username}",
        0xFFA500
    )
    return jsonify({"success": True})

def edit_tokens(username: str, tokens: float) -> Tuple[dict, int]:
    try:
        tokens = float(tokens)
    except ValueError:
        return jsonify({"error": "Invalid tokens value", "code": "invalid-tokens-value"}), 400
    
    if not Collections['users'].find_one({"username": username}):
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404
    
    Collections['users'].update_one({"username": username}, {"$set": {"tokens": tokens}})
    send_discord_notification(
        "Tokens Edited",
        f"Admin {request.username} set {username}'s tokens to {tokens}",
        0xFFA500
    )
    return jsonify({"success": True})

def edit_exp(username: str, exp: float) -> Tuple[dict, int]:
    try:
        exp = float(exp)
        if exp < 0:
            return jsonify({"error": "Exp cannot be negative", "code": "cannot-be-negative"}), 400
    except ValueError:
        return jsonify({"error": "Invalid exp value", "code": "invalid-value"}), 400
    
    if not Collections['users'].find_one({"username": username}):
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404
    
    set_exp(username, exp)
    send_discord_notification(
        "Experience Edited",
        f"Admin {request.username} set {username}'s exp to {exp}",
        0xFFA500
    )
    return jsonify({"success": True})

def edit_level(username: str, level: int) -> Tuple[dict, int]:
    try:
        level = int(level)
        if level < 1:
            return jsonify({"error": "Level cannot be less than 1", "code": "cannot-be-negative"}), 400
    except ValueError:
        return jsonify({"error": "Invalid level value", "code": "invalid-value"}), 400
    
    if not Collections['users'].find_one({"username": username}):
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404
    
    set_level(username, level)
    send_discord_notification(
        "Level Edited",
        f"Admin {request.username} set {username}'s level to {level}",
        0xFFA500
    )
    return jsonify({"success": True})

def add_admin(username: str) -> Tuple[dict, int]:
    if not Collections['users'].find_one({"username": username}):
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404
    
    Collections['users'].update_one({"username": username}, {"$set": {"type": "admin"}})
    send_discord_notification(
        "Admin Added",
        f"Admin {request.username} added {username} as admin",
        0xFFA500
    )
    return jsonify({"success": True})

def remove_admin(username: str) -> Tuple[dict, int]:
    if not Collections['users'].find_one({"username": username}):
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404
    
    Collections['users'].update_one({"username": username}, {"$set": {"type": "user"}})
    send_discord_notification(
        "Admin Removed",
        f"Admin {request.username} removed {username} as admin",
        0xFFA500
    )
    return jsonify({"success": True})

def add_mod(username: str) -> Tuple[dict, int]:
    if not Collections['users'].find_one({"username": username}):
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404
    
    Collections['users'].update_one({"username": username}, {"$set": {"type": "mod"}})
    send_discord_notification(
        "Mod Added",
        f"Admin {request.username} added {username} as mod",
        0xFFA500
    )
    return jsonify({"success": True})

def remove_mod(username: str) -> Tuple[dict, int]:
    if not Collections['users'].find_one({"username": username}):
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404
    
    Collections['users'].update_one({"username": username}, {"$set": {"type": "user"}})
    send_discord_notification(
        "Mod Removed",
        f"Admin {request.username} removed {username} as mod",
        0xFFA500
    )
    return jsonify({"success": True})

def edit_item(item_id: str, new_name: str, new_icon: str, new_rarity: str) -> Tuple[dict, int]:
    item = Collections['items'].find_one({"id": item_id})
    if not item:
        return jsonify({"error": "Item not found", "code": "item-not-found"}), 404
    
    updates = {}
    if new_name:
        parts = split_name(new_name)
        updates.update({
            "name.adjective": html.escape(parts["adjective"].strip()),
            "name.material": html.escape(parts["material"].strip()),
            "name.noun": html.escape(parts["noun"].strip()),
            "name.suffix": html.escape(parts["suffix"].strip()),
            "name.number": html.escape(parts["number"].strip())
        })
    if new_icon:
        updates["name.icon"] = html.escape(new_icon.strip())
    if new_rarity:
        rarity = float(new_rarity)
        updates["rarity"] = rarity
        updates["level"] = get_level(rarity)
    
    if updates:
        Collections['items'].update_one({"id": item_id}, {"$set": updates})
        updated_item = Collections['items'].find_one({"id": item_id}, {"_id": 0})
        item_name = " ".join([
            updated_item["name"]["adjective"], updated_item["name"]["material"],
            updated_item["name"]["noun"], updated_item["name"]["suffix"],
            f"#{updated_item['name']['number']}"
        ]).strip()
        updates_str = ", ".join([f"{k}: {v}" for k, v in updates.items()])
        send_discord_notification(
            "Item Edited",
            f"Admin {request.username} edited {item_name} (ID: {item_id}). Changes: {updates_str}",
            0xFFA500
        )
    return jsonify({"success": True})

def delete_item(item_id: str) -> Tuple[dict, int]:
    item = Collections['items'].find_one({"id": item_id})
    if not item:
        return jsonify({"error": "Item not found", "code": "item-not-found"}), 404
    
    Collections['users'].update_one({"username": item["owner"]}, {"$pull": {"items": item_id}})
    Collections['items'].delete_one({"id": item_id})
    send_discord_notification(
        "Item Deleted",
        f"Admin {request.username} deleted item {item_id}",
        0xFF0000
    )
    return jsonify({"success": True})

def ban_user(username: str, length: str, reason: str) -> Tuple[dict, int]:
    user = Collections['users'].find_one({"username": username})
    if not user:
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404
    if user.get("type") == "admin":
        return jsonify({"error": "Cannot ban admin", "code": "cannot-ban-admin"}), 403
    
    end_time = parse_time(length)
    Collections['users'].update_one(
        {"username": username},
        {"$set": {"banned_until": end_time, "banned_reason": reason, "banned": True}}
    )
    send_discord_notification(
        "User Banned",
        f"Admin {request.username} banned {username} for {length}. Reason: {reason}",
        0xFF0000
    )
    return jsonify({"success": True})

def unban_user(username: str) -> Tuple[dict, int]:
    if not Collections['users'].find_one({"username": username}):
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404
    
    Collections['users'].update_one(
        {"username": username},
        {"$set": {"banned_until": None, "banned_reason": None, "banned": False}}
    )
    send_discord_notification(
        "User Unbanned",
        f"Admin {request.username} unbanned {username}",
        0xFFA500
    )
    return jsonify({"success": True})

def mute_user(username: str, length: str, notify: bool = True) -> Tuple[dict, int]:
    if not Collections['users'].find_one({"username": username}):
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404
    
    end_time = parse_time(length)
    Collections['users'].update_one(
        {"username": username},
        {"$set": {"muted_until": end_time, "muted": True}}
    )
    if notify:
        send_discord_notification(
            "User Muted",
            f"Admin/Mod {request.username} muted {username} for {length}",
            0xFFA500
        )
    return jsonify({"success": True})

def unmute_user(username: str, notify: bool = True) -> Tuple[dict, int]:
    if not Collections['users'].find_one({"username": username}):
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404
    
    Collections['users'].update_one(
        {"username": username}, {"$set": {"muted": False, "muted_until": None}}
    )
    if notify:
        send_discord_notification(
            "User Unmuted",
            f"Admin/Mod {request.username} unmuted {username}",
            0xFFA500
        )
    return jsonify({"success": True})

def fine_user(username: str, amount: int) -> Tuple[dict, int]:
    if not Collections['users'].find_one({"username": username}):
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404
    
    Collections['users'].update_one({"username": username}, {"$inc": {"tokens": -amount}})
    send_discord_notification(
        "User Fined",
        f"Admin {request.username} fined {username} {amount} tokens",
        0xFFA500
    )
    return jsonify({"success": True})

def delete_message(message_id: str) -> Tuple[dict, int]:
    if not message_id:
        return jsonify({"error": "Missing message_id", "code": "missing-parameters"}), 400
    
    Collections['messages'].delete_one({"id": message_id})
    send_discord_notification(
        "Message Deleted",
        f"Mod/Admin {request.username} deleted message {message_id}",
        0xFF0000
    )
    return jsonify({"success": True})

# Routes
@app.route("/")
def index():
    return redirect("https://economix.lol/")

@app.route("/api/register", methods=["POST"])
def register_endpoint():
    data = request.get_json()
    return register(data.get("username"), data.get("password"), request.remote_addr)

@app.route("/api/login", methods=["POST"])
def login_endpoint():
    data = request.get_json()
    return login(data.get("username"), data.get("password"), request.remote_addr,
                data.get("code"), data.get("token"))

@app.route("/api/setup_2fa", methods=["POST"])
@requires_unbanned
def setup_2fa_endpoint():
    user = Collections['users'].find_one({"username": request.username})
    if user.get("2fa_enabled", False):
        return jsonify({"error": "2FA already enabled", "code": "2fa-already-enabled"}), 400
    
    secret = user.get("2fa_secret") or pyotp.random_base32(32)
    code = user.get("2fa_code") or str(uuid4())
    Collections['users'].update_one(
        {"username": request.username},
        {"$set": {"2fa_secret": secret, "2fa_code": code}}
    )
    totp = pyotp.TOTP(secret)
    uri = totp.provisioning_uri(
        name=request.username, issuer_name="Economix",
        image="https://economix.proplayer919.dev/brand/logo.png"
    )
    send_discord_notification("2FA enabled", f"Username: {request.username}")
    return jsonify({"success": True, "provisioning_uri": uri, "backup_code": code})

@app.route("/api/2fa_qrcode", methods=["GET"])
@requires_unbanned
def get_2fa_qrcode_endpoint():
    user = Collections['users'].find_one({"username": request.username})
    if "2fa_secret" not in user:
        return jsonify({"error": "2FA not enabled", "code": "2fa-not-enabled"}), 400
    totp = pyotp.TOTP(user["2fa_secret"])
    uri = totp.provisioning_uri(
        name=request.username, issuer_name="Economix",
        image="https://economix.proplayer919.dev/brand/logo.png"
    )
    img = qrcode.make(uri)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

@app.route("/api/verify_2fa", methods=["POST"])
@requires_unbanned
def verify_2fa_endpoint():
    data = request.get_json()
    user = Collections['users'].find_one({"username": request.username})
    if "2fa_secret" not in user:
        return jsonify({"error": "2FA not setup", "code": "2fa-not-setup"}), 400
    totp = pyotp.TOTP(user["2fa_secret"])
    if not totp.verify(data.get("code")):
        return jsonify({"error": "Invalid 2FA token", "code": "invalid-2fa-token"}), 401
    Collections['users'].update_one(
        {"username": request.username}, {"$set": {"2fa_enabled": True}}
    )
    return jsonify({"success": True})

@app.route("/api/disable_2fa", methods=["POST"])
@requires_unbanned
def disable_2fa_endpoint():
    Collections['users'].update_one(
        {"username": request.username},
        {"$set": {"2fa_enabled": False, "2fa_secret": None, "2fa_code": None}}
    )
    send_discord_notification("2FA disabled", f"Username: {request.username}")
    return jsonify({"success": True})

@app.route("/api/account", methods=["GET"])
def account_endpoint():
    update_account(request.username)
    user = Collections['users'].find_one({"username": request.username})
    items = list(Collections['items'].find({"id": {"$in": user["items"]}}, {"_id": 0}))
    pets = list(Collections['pets'].find({"id": {"$in": user["pets"]}}, {"_id": 0}))
    return jsonify({
        "username": user["username"], "type": user.get("type", "user"),
        "tokens": user["tokens"], "items": items, "last_item_time": user["last_item_time"],
        "last_mine_time": user["last_mine_time"], "banned_until": user.get("banned_until"),
        "banned_reason": user.get("banned_reason"), "banned": user.get("banned"),
        "muted": user.get("muted"), "muted_until": user.get("muted_until"),
        "exp": user.get("exp"), "level": user.get("level"), "history": user.get("history"),
        "2fa_enabled": user.get("2fa_enabled"), "pets": pets
    })

@app.route("/api/delete_account", methods=["POST"])
@requires_unbanned
def delete_account_endpoint():
    user = Collections['users'].find_one({"username": request.username})
    if not user:
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404
    Collections['items'].delete_many({"owner": request.username})
    Collections['users'].delete_one({"username": request.username})
    send_discord_notification("User deleted", f"Username: {request.username}")
    return jsonify({"success": True})

@app.route("/api/create_item", methods=["POST"])
@requires_unbanned
def create_item_endpoint():
    now = time.time()
    user = Collections['users'].find_one({"username": request.username}, {"_id": 0})
    if not user:
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404
    if now - user["last_item_time"] < ITEM_CREATE_COOLDOWN:
        return jsonify({
            "error": "Cooldown active",
            "remaining": ITEM_CREATE_COOLDOWN - (now - user["last_item_time"]),
            "code": "cooldown-active"
        }), 429
    if user["tokens"] < 10:
        return jsonify({"error": "Not enough tokens", "code": "not-enough-tokens"}), 402
    
    new_item = generate_item(request.username)
    Collections['items'].insert_one(new_item)
    Collections['users'].update_one(
        {"username": request.username},
        {"$push": {"items": new_item["id"]}, "$set": {"last_item_time": now}, "$inc": {"tokens": -10}}
    )
    Collections['users'].update_one(
        {"username": request.username},
        {"$push": {"history": {"item_id": new_item["id"], "action": "create", "timestamp": now}}}
    )
    add_exp(request.username, 10)
    
    item_name = " ".join([
        new_item["name"]["adjective"], new_item["name"]["material"], new_item["name"]["noun"],
        new_item["name"]["suffix"], f"#{new_item['name']['number']}"
    ]).strip()
    send_discord_notification(
        "New Item Created",
        f"User {request.username} created: {item_name} (Rarity: {new_item['rarity']})"
    )
    return jsonify({k: v for k, v in new_item.items() if k not in ["_id", "item_secret"]})

@app.route("/api/buy_pet", methods=["POST"])
@requires_unbanned
def buy_pet_endpoint():
    user = Collections["users"].find_one({"username": request.username})
    if not user:
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404
    if len(user.get("pets", [])) >= 1:
        current_pet = Collections["pets"].find_one({"id": user["pets"][0]})
        if current_pet["alive"]:
            return (
                jsonify(
                    {"error": "Already has a live pet", "code": "user-already-has-pet"}
                ),
                400,
            )
        # Remove dead pet
        Collections["users"].update_one(
            {"username": request.username}, {"$pull": {"pets": current_pet["id"]}}
        )
        price = current_pet["base_price"] * 2  # Double price for repurchase
    else:
        price = 100  # Initial price

    if user["tokens"] < price:
        return jsonify({"error": "Not enough tokens", "code": "not-enough-tokens"}), 402

    pet = generate_pet(request.username, price)
    Collections["pets"].insert_one(pet)
    Collections["users"].update_one(
        {"username": request.username},
        {"$inc": {"tokens": -price}, "$push": {"pets": pet["id"]}},
    )
    send_discord_notification(
        "New Pet Bought",
        f"User {request.username} bought pet: {pet['name']} for {price} tokens",
    )
    return jsonify(pet)

@app.route("/api/feed_pet", methods=["POST"])
@requires_unbanned
def feed_pet_endpoint():
    data = request.get_json()
    pet_id = data.get("pet_id")
    user = Collections["users"].find_one({"username": request.username})
    if not user:
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404
    pet = Collections["pets"].find_one({"id": pet_id})
    if not pet or not pet["alive"]:
        return jsonify({"error": "Pet not found or dead", "code": "pet-not-found"}), 404
    if user["tokens"] < 10:
        return jsonify({"error": "Not enough tokens", "code": "not-enough-tokens"}), 402

    Collections["pets"].update_one(
        {"id": pet_id}, {"$set": {"last_fed": int(time.time())}}
    )
    Collections["users"].update_one(
        {"username": request.username}, {"$inc": {"tokens": -10}}
    )
    level_up_pet(pet_id, 3)  # Gain 3 exp per feeding
    update_pet(pet_id)  # Refresh status and benefits
    send_discord_notification(
        "Pet Fed", f"User {request.username} fed pet: {pet['name']}"
    )
    return jsonify({"success": True})

@app.route("/api/mine_tokens", methods=["POST"])
@requires_unbanned
def mine_tokens_endpoint():
    now = time.time()
    user = Collections['users'].find_one({"username": request.username}, {"_id": 0})
    if not user:
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404
    if now - user["last_mine_time"] < TOKEN_MINE_COOLDOWN:
        return jsonify({
            "error": "Cooldown active",
            "remaining": TOKEN_MINE_COOLDOWN - (now - user["last_mine_time"]),
            "code": "cooldown-active"
        }), 429
    
    tokens = random.randint(5, 10)
    Collections['users'].update_one(
        {"username": request.username},
        {"$inc": {"tokens": tokens}, "$set": {"last_mine_time": now}}
    )
    Collections['users'].update_one(
        {"username": request.username},
        {"$push": {"history": {"item_id": None, "action": "mine", "timestamp": now}}}
    )
    add_exp(request.username, 5)
    send_discord_notification(
        "Tokens Mined",
        f"User {request.username} mined {tokens} tokens"
    )
    return jsonify({"success": True, "tokens": tokens})

@app.route("/api/market", methods=["GET"])
@requires_unbanned
def market_endpoint():
    items = Collections['items'].find(
        {"for_sale": True, "owner": {"$ne": request.username}}, {"_id": 0, "item_secret": 0}
    )
    return jsonify(list(items))

@app.route("/api/sell_item", methods=["POST"])
@requires_unbanned
def sell_item_endpoint():
    data = request.get_json()
    try:
        price = float(data.get("price"))
        if not MIN_ITEM_PRICE <= price <= MAX_ITEM_PRICE:
            raise ValueError
    except ValueError:
        return jsonify({"error": f"Invalid price ({MIN_ITEM_PRICE}-{MAX_ITEM_PRICE})"}), 400
    
    item = Collections['items'].find_one({"id": data.get("item_id"), "owner": request.username}, {"_id": 0})
    if not item:
        return jsonify({"error": "Item not found", "code": "item-not-found"}), 404
    
    update_data = {"for_sale": not item["for_sale"], "price": price if not item["for_sale"] else 0}
    Collections['items'].update_one({"id": data.get("item_id")}, {"$set": update_data})
    Collections['users'].update_one(
        {"username": request.username},
        {"$push": {"history": {"item_id": data.get("item_id"), "action": "sell", "timestamp": time.time()}}}
    )
    
    item_name = " ".join([
        item["name"]["adjective"], item["name"]["material"], item["name"]["noun"],
        item["name"]["suffix"], f"#{item['name']['number']}"
    ]).strip()
    send_discord_notification(
        "Item Listed" if update_data["for_sale"] else "Item Unlisted",
        f"User {request.username} {'listed' if update_data['for_sale'] else 'unlisted'} {item_name} {'for ' + str(price) + ' tokens' if update_data['for_sale'] else ''}",
        0xFFFF00
    )
    return jsonify({"success": True})

@app.route("/api/buy_item", methods=["POST"])
@requires_unbanned
def buy_item_endpoint():
    data = request.get_json()
    item = Collections['items'].find_one({"id": data.get("item_id"), "for_sale": True}, {"_id": 0})
    if not item:
        return jsonify({"error": "Item not available", "code": "item-not-found"}), 404
    if item["owner"] == request.username:
        return jsonify({"error": "Cannot buy own item", "code": "cannot-buy-own-item"}), 400
    
    buyer = Collections['users'].find_one({"username": request.username}, {"_id": 0})
    if buyer["tokens"] < item["price"]:
        return jsonify({"error": "Not enough tokens", "code": "not-enough-tokens"}), 402
    
    with client.start_session() as session:
        with session.start_transaction():
            Collections['users'].update_one(
                {"username": request.username}, {"$inc": {"tokens": -item["price"]}}, session=session
            )
            Collections['users'].update_one(
                {"username": item["owner"]}, {"$inc": {"tokens": item["price"]}}, session=session
            )
            Collections['users'].update_one(
                {"username": item["owner"]}, {"$pull": {"items": data.get("item_id")}}, session=session
            )
            Collections['users'].update_one(
                {"username": request.username}, {"$push": {"items": data.get("item_id")}}, session=session
            )
            Collections['items'].update_one(
                {"id": data.get("item_id")},
                {"$set": {"owner": request.username, "for_sale": False, "price": 0}},
                session=session
            )
    
    Collections['users'].update_one(
        {"username": request.username},
        {"$push": {"history": {"item_id": data.get("item_id"), "action": "buy", "timestamp": time.time()}}}
    )
    Collections['users'].update_one(
        {"username": item["owner"]},
        {"$push": {"history": {"item_id": data.get("item_id"), "action": "sell_complete", "timestamp": time.time()}}}
    )
    
    meta = Collections['item_meta'].find_one({"id": item["meta_id"]})
    if meta:
        meta["price_history"].append({"timestamp": time.time(), "price": item["price"]})
        Collections['item_meta'].update_one({"id": item["meta_id"]}, {"$set": meta})
    
    add_exp(request.username, 5)
    add_exp(item["owner"], 5)
    
    item_name = " ".join([
        item["name"]["adjective"], item["name"]["material"], item["name"]["noun"],
        item["name"]["suffix"], f"#{item['name']['number']}"
    ]).strip()
    send_discord_notification(
        "Item Purchased",
        f"User {request.username} bought {item_name} from {item['owner']} for {item['price']} tokens",
        0x0000FF
    )
    return jsonify({"success": True})

@app.route("/api/take_item", methods=["POST"])
@requires_unbanned
def take_item_endpoint():
    data = request.get_json()
    item = Collections['items'].find_one({"item_secret": data.get("item_secret")})
    if not item:
        return jsonify({"error": "Invalid secret", "code": "invalid-secret"}), 404
    
    with client.start_session() as session:
        with session.start_transaction():
            Collections['users'].update_one(
                {"username": item["owner"]}, {"$pull": {"items": item["id"]}}, session=session
            )
            Collections['users'].update_one(
                {"username": request.username}, {"$push": {"items": item["id"]}}, session=session
            )
            Collections['items'].update_one(
                {"item_secret": data.get("item_secret")},
                {"$set": {"owner": request.username, "for_sale": False, "price": 0}},
                session=session
            )
    
    Collections['users'].update_one(
        {"username": request.username},
        {"$push": {"history": {"item_id": item["id"], "action": "take", "timestamp": time.time()}}}
    )
    Collections['users'].update_one(
        {"username": item["owner"]},
        {"$push": {"history": {"item_id": item["id"], "action": "taken_from", "timestamp": time.time()}}}
    )
    return jsonify({"success": True})

@app.route("/api/leaderboard", methods=["GET"])
@requires_unbanned
def leaderboard_endpoint():
    pipeline = [
        {"$match": {"banned": {"$ne": True}}},
        {"$sort": {"tokens": DESCENDING}},
        {"$limit": 10},
        {"$project": {"_id": 0, "username": 1, "tokens": 1}}
    ]
    results = list(Collections['users'].aggregate(pipeline))
    
    def ordinal(n):
        return "%d%s" % (n, "tsnrhtdd"[((n // 10 % 10 != 1) * (n % 10 < 4) * n % 10) :: 4])
    
    for i, item in enumerate(results):
        item["place"] = ordinal(i + 1)
    return jsonify({"leaderboard": results})

@app.route("/api/send_message", methods=["POST"])
@requires_unbanned
def send_message_endpoint():
    data = request.get_json()
    return send_message(data.get("room", "global"), data.get("message"), request.username, request.remote_addr)

@app.route("/api/get_messages", methods=["GET"])
@requires_unbanned
def get_messages_endpoint():
    room = request.args.get("room", "global")
    if not Collections['users'].find_one({"username": request.username}):
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404
    if not room:
        return jsonify({"error": "Missing room parameter", "code": "missing-parameters"}), 400
    messages = Collections['messages'].find({"room": room}, {"_id": 0}).sort("timestamp", ASCENDING)
    return jsonify({"messages": list(messages)})

@app.route("/api/get_banner", methods=["GET"])
@requires_unbanned
def get_banner_endpoint():
    banner = Collections['misc'].find_one({"type": "banner"}, {"_id": 0})
    return jsonify({"banner": banner})

@app.route("/api/stats", methods=["GET"])
def stats_endpoint():
    accounts = list(Collections['users'].find())
    items = list(Collections['items'].find())
    mods = list(Collections['users'].find({"type": "mod"}))
    admins = list(Collections['users'].find({"type": "admin"}))
    users = list(Collections['users'].find({"type": "user"}))
    total_tokens = sum(user["tokens"] for user in accounts)
    
    return jsonify({
        "stats": {
            "total_tokens": total_tokens,
            "total_accounts": len(accounts),
            "total_items": len(items),
            "total_mods": len(mods),
            "total_admins": len(admins),
            "total_users": len(users)
        }
    })

@app.route("/api/recycle_item", methods=["POST"])
@requires_unbanned
def recycle_endpoint():
    data = request.get_json()
    item_id = data["item_id"]
    item = Collections['items'].find_one({"id": item_id})
    if not item:
        return jsonify({"error": "Item not found"}), 404
    if item["owner"] != request.username:
        return jsonify({"error": "Unauthorized"}), 403
    
    user = Collections['users'].find_one({"username": request.username})
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    Collections['users'].update_one(
        {"username": request.username},
        {"$pull": {"items": item_id}, "$inc": {"tokens": 5}}
    )
    Collections['items'].delete_one({"id": item_id})
    return jsonify({"success": True})

@app.route("/api/reset_cooldowns", methods=["POST"])
@requires_admin
def reset_cooldowns_endpoint():
    return reset_cooldowns(request.username)

@app.route("/api/edit_tokens", methods=["POST"])
@requires_admin
def edit_tokens_endpoint():
    data = request.get_json()
    return edit_tokens(data.get("username") or request.username, data.get("tokens"))

@app.route("/api/edit_exp", methods=["POST"])
@requires_admin
def edit_exp_endpoint():
    data = request.get_json()
    return edit_exp(data.get("username") or request.username, data.get("exp"))

@app.route("/api/edit_level", methods=["POST"])
@requires_admin
def edit_level_endpoint():
    data = request.get_json()
    return edit_level(data.get("username") or request.username, data.get("level"))

@app.route("/api/add_admin", methods=["POST"])
@requires_admin
def add_admin_endpoint():
    data = request.get_json()
    return add_admin(data.get("username"))

@app.route("/api/remove_admin", methods=["POST"])
@requires_admin
def remove_admin_endpoint():
    data = request.get_json()
    return remove_admin(data.get("username"))

@app.route("/api/add_mod", methods=["POST"])
@requires_admin
def add_mod_endpoint():
    data = request.get_json()
    return add_mod(data.get("username"))

@app.route("/api/remove_mod", methods=["POST"])
@requires_admin
def remove_mod_endpoint():
    data = request.get_json()
    return remove_mod(data.get("username"))

@app.route("/api/edit_item", methods=["POST"])
@requires_admin
def edit_item_endpoint():
    data = request.get_json()
    return edit_item(data.get("item_id"), data.get("new_name"), data.get("new_icon"), data.get("new_rarity"))

@app.route("/api/delete_item", methods=["POST"])
@requires_admin
def delete_item_endpoint():
    data = request.get_json()
    return delete_item(data.get("item_id"))

@app.route("/api/ban_user", methods=["POST"])
@requires_admin
def ban_user_endpoint():
    data = request.get_json()
    return ban_user(data.get("username"), data.get("length"), data.get("reason"))

@app.route("/api/unban_user", methods=["POST"])
@requires_admin
def unban_user_endpoint():
    data = request.get_json()
    return unban_user(data.get("username"))

@app.route("/api/fine_user", methods=["POST"])
@requires_admin
def fine_user_endpoint():
    data = request.get_json()
    return fine_user(data.get("username"), data.get("amount"))

@app.route("/api/mute_user", methods=["POST"])
@requires_mod
def mute_user_endpoint():
    data = request.get_json()
    return mute_user(data.get("username"), data.get("length"))

@app.route("/api/unmute_user", methods=["POST"])
@requires_mod
def unmute_user_endpoint():
    data = request.get_json()
    return unmute_user(data.get("username"))

@app.route("/api/users", methods=["GET"])
@requires_mod
def users_endpoint():
    return get_users()

@app.route("/api/delete_message", methods=["POST"])
@requires_mod
def delete_message_endpoint():
    data = request.get_json()
    return delete_message(data.get("message_id"))

@app.route("/api/set_banner", methods=["POST"])
@requires_admin
def set_banner_endpoint():
    data = request.get_json()
    Collections['misc'].delete_many({"type": "banner"})
    Collections['misc'].insert_one({"type": "banner", "value": data.get("banner")})
    send_discord_notification(
        "Banner Updated",
        f"Admin {request.username} set banner to {data.get('banner')}",
        0x00FF00
    )
    return jsonify({"success": True})

@app.route("/api/get_banned", methods=["GET"])
@requires_admin
def get_banned_endpoint():
    banned = Collections['users'].find({"banned": True}, {"_id": 0})
    return jsonify({"banned_users": [user["username"] for user in banned]})

@app.route("/api/delete_user", methods=["POST"])
@requires_admin
def delete_user_endpoint():
    data = request.get_json()
    username = data.get("username")
    user = Collections['users'].find_one({"username": username})
    if not user:
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404
    Collections['items'].delete_many({"owner": username})
    Collections['users'].delete_one({"username": username})
    send_discord_notification("User deleted", f"Admin {request.username} deleted user {username}", 0xFF0000)
    return jsonify({"success": True})