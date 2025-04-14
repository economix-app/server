import os
import json
import time
import random
import logging
from uuid import uuid4
from threading import Thread
from typing import Dict, Optional, Tuple, Callable, List
import traceback
import sys

from flask import Flask, request, jsonify, send_file, redirect
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from hashlib import sha256
from functools import wraps
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError, PyMongoError
import re
import html
import pyotp
import qrcode
import io
from better_profanity import profanity
import requests
from logging.handlers import RotatingFileHandler
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import stripe
import importlib.util

# Constants
ITEM_CREATE_COOLDOWN = 60  # 1 minute
TOKEN_MINE_COOLDOWN = 180  # 3 minutes
MAX_ITEM_PRICE = 1000 * 1000 * 100  # 1 million
MIN_ITEM_PRICE = 1

DEBUG_MODE = os.environ.get("FLASK_DEBUG", "true").lower() == "true"

# Application Setup
app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get("FLASK_SECRET_KEY", "1234"),
)
CORS(app, origins=os.environ.get("CORS_ORIGINS", "").split(","))

# Logging Configuration
handler = RotatingFileHandler("app.log", maxBytes=10 * 1024 * 1024, backupCount=5)
handler.setFormatter(
    logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]"
    )
)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

log_file = open("app.log", "r")
log_file.seek(0, 2)  # Seek to the end of the file
active_queues = set()  # Track active SSE connections

# Stripe configuration
STRIPE_PUBLISHABLE_KEY = os.environ.get("STRIPE_PUBLISHABLE_KEY")
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET")

stripe.api_key = STRIPE_SECRET_KEY

PRICE_IDS = {
    "gems_500": "price_1RDKn6DOJm6z6Mj6ZUUdJV5j",
    "gems_1000": "price_1RDKnfDOJm6z6Mj61nxE69Fd",
    "gems_2500": "price_1RDKoCDOJm6z6Mj6TYoqrNAf",
    "gems_5000": "price_1RDKojDOJm6z6Mj6kG1LiidX",
    "pro_monthly": "price_1R5f6MDOJm6z6Mj6KHCNb49F",
    "pro_yearly": "price_1R5fCwDOJm6z6Mj659CUm8K7",
    "pro_plus_monthly": "price_1R5f9WDOJm6z6Mj61WRCGsja",
    "pro_plus_yearly": "price_1R5fBxDOJm6z6Mj6g6A5v5bZ",
}


class LogHandler(FileSystemEventHandler):
    """Handle log file events detected by watchdog."""

    def on_modified(self, event):
        """Read new lines when the log file is modified."""
        if event.src_path == "app.log":
            while True:
                line = log_file.readline()
                if not line:
                    break
                for q in active_queues:
                    q.put(line)

    def on_created(self, event):
        """Reopen the log file when a new one is created (e.g., after rotation)."""
        if event.src_path == "app.log":
            global log_file
            log_file.close()
            log_file = open("app.log", "r")
            log_file.seek(0, 2)


# Initialize and start the watchdog observer
observer = Observer()
observer.schedule(LogHandler(), path=".", recursive=False)
observer.start()

# Database Setup
client = MongoClient(os.environ.get("MONGODB_URI"), maxPoolSize=50)
db = client[os.environ.get("MONGODB_DB")]
Collections = {
    "users": db.users,
    "items": db.items,
    "messages": db.messages,
    "item_meta": db.item_meta,
    "misc": db.misc,
    "pets": db.pets,
    "account_creation_attempts": db.account_creation_attempts,
    "message_attempts": db.message_attempts,
    "blocked_ips": db.blocked_ips,
    "failed_logins": db.failed_logins,
    "user_history": db.user_history,
    "creator_codes": db.creator_codes,
    "companies": db.companies,
    "auctions": db.auctions,
    "trades": db.trades,
    "reports": db.reports,
    "pending_subscriptions": db.pending_subscriptions,
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
    "MIN_ACCOUNT_AGE": 3600,
    "SUBNET_BLOCKING": True,
    "TOKEN_TRANSFER_THRESHOLD": 1000,  # Max tokens transferable in a single transaction
    "TOKEN_TRANSFER_TIME_WINDOW": 3600,  # 1 hour
    "NEW_ACCOUNT_TOKEN_TRANSFER_LIMIT": 20,  # Max tokens transferable by new accounts
    "NEW_ACCOUNT_AGE_LIMIT": 86400,  # 24 hours
    "EXPLOIT_DETECTION_TIME_WINDOW": 3600,  # 1 hour
    "EXPLOIT_DETECTION_THRESHOLD": 3,  # Max suspicious actions in the time window
}

# Cosmetics
COSMETICS = {
    # Messageplates
    "bamboo-forest": {
        "type": "messageplate",
        "name": "Bamboo Forest",
        "price": 300,
        "id": "bamboo-forest",
    },
    "bonzai-bliss": {
        "type": "messageplate",
        "name": "Bonzai Bliss",
        "price": 400,
        "id": "bonzai-bliss",
    },
    "cherry-blossom": {
        "type": "messageplate",
        "name": "Cherry Blossom",
        "price": 450,
        "id": "cherry-blossom",
    },
    "full-moon": {
        "type": "messageplate",
        "name": "Full Moon",
        "price": 350,
        "id": "full-moon",
    },
    "lantern-festival": {
        "type": "messageplate",
        "name": "Lantern Festival",
        "price": 500,
        "id": "lantern-festival",
    },
    "starry-night": {
        "type": "messageplate",
        "name": "Starry Night",
        "price": 450,
        "id": "starry-night",
    },
    "tokyo-tower": {
        "type": "messageplate",
        "name": "Tokyo Tower",
        "price": 500,
        "id": "tokyo-tower",
    },
    "sun-rays": {
        "type": "messageplate",
        "name": "Sun Rays",
        "price": 350,
        "id": "sun-rays",
    },
    "planets-align": {
        "type": "messageplate",
        "name": "Planets Align",
        "price": 300,
        "id": "planets-align",
    },
    "cityscape-dreams": {
        "type": "messageplate",
        "name": "Cityscape Dreams",
        "price": 400,
        "id": "cityscape-dreams",
    },
    "lightning-storm": {
        "type": "messageplate",
        "name": "Lightning Storm",
        "price": 500,
        "id": "lightning-storm",
    },
    # Nameplates
    "gold": {
        "type": "nameplate",
        "name": "Gold",
        "price": None,
        "id": "gold",
    },
}


@app.errorhandler(500)
def internal_server_error(error):
    # Log the full exception details
    exc_type, exc_value, exc_traceback = sys.exc_info()
    stack_trace = "".join(
        traceback.format_exception(exc_type, exc_value, exc_traceback)
    )

    error_message = str(error) or "An unexpected error occurred"
    app.logger.error(
        f"500 Internal Server Error: {error_message}\nStack Trace:\n{stack_trace}"
    )

    # Detailed response for debugging (only in DEBUG mode or if explicitly enabled)
    if DEBUG_MODE:
        response = {
            "error": "Internal Server Error",
            "code": "internal-server-error",
            "message": error_message,
            "details": {
                "exception": str(exc_type.__name__),
                "description": str(exc_value),
                "stack_trace": stack_trace.splitlines(),
                "request": {
                    "method": request.method,
                    "url": request.url,
                    "headers": dict(request.headers),
                    "body": request.get_data(as_text=True) if request.data else None,
                    "remote_addr": request.remote_addr,
                },
            },
            "timestamp": int(time.time()),
        }
    else:
        # Minimal response for production
        response = {
            "error": "Internal Server Error",
            "code": "internal-server-error",
            "message": "Something went wrong on the server. Please try again later.",
            "timestamp": int(time.time()),
        }

    return jsonify(response), 500


@app.errorhandler(Exception)
def handle_unhandled_exception(e):
    # Redirect all unhandled exceptions to the 500 handler
    return internal_server_error(e)


# Load Word Lists
def load_word_lists():
    global ADJECTIVES, MATERIALS, NOUNS, SUFFIXES, PET_NAMES
    try:
        with open("words/adjectives.json") as f:
            ADJECTIVES = json.load(f)
        with open("words/materials.json") as f:
            MATERIALS = json.load(f)
        with open("words/nouns.json") as f:
            NOUNS = json.load(f)
        with open("words/suffixes.json") as f:
            SUFFIXES = json.load(f)
        with open("words/pet_names.json") as f:
            PET_NAMES = json.load(f)
        app.logger.info("Loaded item generation word lists successfully")
    except Exception as e:
        app.logger.critical(f"Failed to load word lists: {str(e)}")
        raise


load_word_lists()
profanity.load_censor_words()


# Index Creation
def create_indexes():
    Collections["users"].create_index([("username", ASCENDING)], unique=True)
    Collections["items"].create_index([("id", ASCENDING), ("owner", ASCENDING)])
    Collections["messages"].create_index(
        [("room", ASCENDING), ("timestamp", ASCENDING)]
    )
    Collections["item_meta"].create_index([("id", ASCENDING)])
    Collections["misc"].create_index([("type", ASCENDING)])
    Collections["pets"].create_index([("id", ASCENDING)], unique=True)
    Collections["account_creation_attempts"].create_index(
        [("timestamp", ASCENDING)],
        expireAfterSeconds=AUTOMOD_CONFIG["ACCOUNT_CREATION_TIME_WINDOW"],
    )
    Collections["message_attempts"].create_index(
        [("timestamp", ASCENDING)],
        expireAfterSeconds=AUTOMOD_CONFIG["MESSAGE_SPAM_TIME_WINDOW"],
    )
    Collections["blocked_ips"].create_index(
        [("blocked_until", ASCENDING)], expireAfterSeconds=0
    )
    Collections["blocked_ips"].create_index([("ip", ASCENDING)])
    Collections["failed_logins"].create_index(
        [("timestamp", ASCENDING)],
        expireAfterSeconds=AUTOMOD_CONFIG["FAILED_LOGIN_WINDOW"],
    )
    Collections["message_attempts"].create_index(
        [("ip", ASCENDING), ("timestamp", ASCENDING)]
    )
    Collections["user_history"].create_index([("username", ASCENDING)])
    Collections["user_history"].create_index([("code", ASCENDING)])
    Collections["companies"].create_index([("name", ASCENDING)], unique=True)
    Collections["companies"].create_index([("owner", ASCENDING)])
    Collections["auctions"].create_index([("item_id", ASCENDING)])
    Collections["auctions"].create_index([("owner", ASCENDING)])
    Collections["trades"].create_index(
        [("offerOwner", ASCENDING), ("requestOwner", ASCENDING)]
    )
    Collections["reports"].create_index([("id", ASCENDING)], unique=True)
    Collections["pending_subscriptions"].create_index(
        [("subscription_id", ASCENDING)], unique=True
    )


create_indexes()


def is_ip_blocked(ip: str) -> bool:
    """Check if an IP or its subnet is blocked"""
    current_time = time.time()
    # Check exact IP match
    if Collections["blocked_ips"].find_one(
        {"ip": ip, "blocked_until": {"$gte": current_time}}
    ):
        return True

    # Check subnet if enabled
    if AUTOMOD_CONFIG["SUBNET_BLOCKING"]:
        subnet = ".".join(ip.split(".")[:3]) + ".0/24"
        if Collections["blocked_ips"].find_one(
            {"subnet": subnet, "blocked_until": {"$gte": current_time}}
        ):
            return True
    return False


def block_ip(ip: str, duration: str, reason: str, subnet: bool = False) -> None:
    """Block an IP address or subnet"""
    end_time = parse_time(duration)
    block_data = {
        "ip": ip,
        "blocked_until": end_time,
        "reason": reason,
        "timestamp": int(time.time()),
    }
    if subnet and AUTOMOD_CONFIG["SUBNET_BLOCKING"]:
        block_data["subnet"] = ".".join(ip.split(".")[:3]) + ".0/24"

    Collections["blocked_ips"].update_one({"ip": ip}, {"$set": block_data}, upsert=True)
    send_discord_notification(
        "IP Blocked",
        f"IP {ip}{' subnet' if subnet else ''} blocked until {time.ctime(end_time)}. Reason: {reason}",
        0xFF0000,
    )


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
        (0.1, "Godlike"),
        (1, "Legendary"),
        (5, "Epic"),
        (10, "Rare"),
        (25, "Uncommon"),
        (50, "Common"),
        (75, "Scrap"),
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
        multipliers = {
            "s": 1,
            "m": 60,
            "h": 3600,
            "d": 86400,
            "w": 604800,
            "y": 31536000,
        }
        duration += value * multipliers.get(unit, 0)
    return int(time.time()) + duration


def get_conversation_id(user1: str, user2: str) -> str:
    return ":".join(sorted([user1, user2]))


def generate_lore(name):
    origins = [
        "forged in dragonfire beneath the scorched skies of the Ashen Peaks",
        "woven from shadow and starlight in the silent halls of Vael'tharan",
        "carved from the bone of a fallen god during the Moonless Age",
        "summoned through a blood ritual lost to all but the voidborn prophets",
        "born in the dreams of the dying world, where time no longer flows",
        "discovered deep within the roots of the World Tree, still pulsing with ancient magic",
    ]

    creators = [
        "the vanished kings of the elder age",
        "sorcerers who bartered their souls for forbidden knowledge",
        "the architects of reality, whose names are carved into the stars",
        "a forgotten civilization buried beneath aeons of silence",
        "a cult sworn to silence, who stitched fate into steel",
        "the last dreamer of a dying god",
    ]

    locations = [
        "the cradle of eternal storm",
        "the sunken city of glass and bone",
        "the forbidden halls of the Mirror Citadel",
        "the edge of the unshaped world",
        "the screaming void between stars",
        "the ruins where gods once wept",
    ]

    powers = [
        "a will that resists time itself",
        "a hunger for truth and madness",
        "the power to unmake reality with a thought",
        "a pulse that echoes in the soul of the bearer",
        "echoes of every life it has ended",
        "the voice of the first silence",
    ]

    fates = [
        "claimed by legend and feared by fate",
        "driven to the brink of divinity or destruction",
        "unraveled by visions of all possible futures",
        "blessed and cursed in equal measure",
        "shackled to an unbreakable prophecy",
        "never to die, yet never to live again",
    ]

    return (
        f"Ancient texts and forgotten songs speak of the {name['material']} {name['noun']}, "
        f"{random.choice(origins)}. It was shaped by {random.choice(creators)}, "
        f"in the heart of {random.choice(locations)}. \n\n"
        f"Legends say it carries {random.choice(powers)}, a force neither light nor dark, "
        f"but something far older. To wield it is to be {random.choice(fates)}.\n\n"
        f"Even now, its story is not finished. It waits, patient and cold, for the next hand bold—or foolish—enough to grasp it."
    )


def send_discord_notification(title: str, description: str, color: int = 0x00FF00):
    webhook_url = os.environ.get("DISCORD_WEBHOOK")
    if not webhook_url:
        app.logger.error("Discord webhook URL not configured")
        return

    def _send():
        data = {
            "embeds": [{"title": title, "description": description, "color": color}]
        }
        response = requests.post(webhook_url, json=data)
        if response.status_code != 204:
            app.logger.error(f"Discord notification failed: {response.status_code}")

        Collections["messages"].insert_one(
            {
                "room": "logs",
                "username": "AutoMod",
                "message": description,
                "timestamp": int(time.time()),
                "type": "system",
            }
        )

    Thread(target=_send).start()


# Authentication Decorators
def requires_admin(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        user = Collections["users"].find_one({"username": request.username})
        if user.get("type") != "admin":
            return (
                jsonify(
                    {"error": "Admin privileges required", "code": "admin-required"}
                ),
                403,
            )
        return f(*args, **kwargs)

    return decorated


def requires_mod(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        user = Collections["users"].find_one({"username": request.username})
        if user.get("type") not in ["admin", "mod"]:
            return (
                jsonify({"error": "Mod privileges required", "code": "mod-required"}),
                403,
            )
        return f(*args, **kwargs)

    return decorated


def requires_unbanned(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        user = Collections["users"].find_one({"username": request.username})
        if user.get("banned_until") and (
            user["banned_until"] > time.time() or user["banned_until"] == 0
        ):
            return jsonify({"error": "You are banned", "code": "banned"}), 403
        return f(*args, **kwargs)

    return decorated


def requires_pro(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        user = Collections["users"].find_one({"username": request.username})
        now = time.time()

        if user.get("override_plan") in ["pro", "proplus"]:
            if (
                user.get("override_plan_expires", 0) == 0
                or user["override_plan_expires"] > now
            ):
                return f(*args, **kwargs)

        subscriptions = user.get("subscriptions", [])
        for sub in subscriptions:
            if sub.get("plan") in ["pro", "proplus"] and sub.get("status") == "active":
                if sub.get("current_period_end", 0) > now:
                    return f(*args, **kwargs)

        return (
            jsonify(
                {"error": "Subscription required", "code": "subscription-required"}
            ),
            403,
        )

    return decorated


def requires_proplus(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        user = Collections["users"].find_one({"username": request.username})
        now = time.time()

        if user.get("override_plan") == "proplus":
            if (
                user.get("override_plan_expires", 0) == 0
                or user["override_plan_expires"] > now
            ):
                return f(*args, **kwargs)

        subscriptions = user.get("subscriptions", [])
        for sub in subscriptions:
            if sub.get("plan") == "proplus" and sub.get("status") == "active":
                if sub.get("current_period_end", 0) > now:
                    return f(*args, **kwargs)

        return (
            jsonify(
                {"error": "Subscription required", "code": "subscription-required"}
            ),
            403,
        )

    return decorated


def has_pro(username):
    user = Collections["users"].find_one({"username": username})
    now = time.time()

    if user.get("override_plan") in ["pro", "proplus"]:
        if (
            user.get("override_plan_expires", 0) == 0
            or user["override_plan_expires"] > now
        ):
            return True

    for sub in user.get("subscriptions", []):
        if sub.get("plan") in ["pro", "proplus"] and sub.get("status") == "active":
            if sub.get("current_period_end", 0) > now:
                return True

    return False


def has_proplus(username):
    user = Collections["users"].find_one({"username": username})
    now = time.time()

    if user.get("override_plan") == "proplus":
        if (
            user.get("override_plan_expires", 0) == 0
            or user["override_plan_expires"] > now
        ):
            return True

    for sub in user.get("subscriptions", []):
        if sub.get("plan") == "proplus" and sub.get("status") == "active":
            if sub.get("current_period_end", 0) > now:
                return True

    return False


def get_plan(username):
    user = Collections["users"].find_one({"username": username})
    now = time.time()

    if user.get("override_plan") in ["pro", "proplus"]:
        if (
            user.get("override_plan_expires", 0) == 0
            or user["override_plan_expires"] > now
        ):
            return user["override_plan"]

    for sub in user.get("subscriptions", []):
        if sub.get("status") == "active" and sub.get("current_period_end", 0) > now:
            return sub.get("plan", "free")

    return "free"


# Middleware
@app.before_request
def authenticate_user():
    public_endpoints = [
        "register_endpoint",
        "login_endpoint",
        "index",
        "stats_endpoint",
        "stripe_webhook",
    ]
    if request.method == "OPTIONS" or request.endpoint in public_endpoints:
        return

    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        return (
            jsonify(
                {
                    "error": "Missing or invalid Authorization header",
                    "code": "invalid-credentials",
                }
            ),
            401,
        )

    token = auth.split(" ")[1]
    user = Collections["users"].find_one({"token": token})
    if not user:
        return jsonify({"error": "Invalid token", "code": "invalid-credentials"}), 401

    request.username = user["username"]
    request.user_type = user.get("type", "user")


# Database Updaters
def update_item(item_id: str):
    item = Collections["items"].find_one({"id": item_id})
    if not item:
        return

    name = item["name"]
    meta_id = (
        item.get("meta_id")
        or sha256(
            f"{name['adjective']}{name['material']}{name['noun']}{name['suffix']}".encode()
        ).hexdigest()
    )

    meta = Collections["item_meta"].find_one({"id": meta_id})
    if not meta:
        rarity = round(random.uniform(0.1, 100), 1)
        lore = generate_lore(name)
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
            "price_history": [],
            "lore": lore,
        }
        Collections["item_meta"].insert_one(meta)

    updates = {
        "meta_id": meta_id,
        "rarity": meta["rarity"],
        "level": meta["level"],
        "lore": meta.get("lore", generate_lore(name)),
    }
    if "history" not in item:
        updates["history"] = []
    Collections["items"].update_one({"id": item_id}, {"$set": updates})


def update_pet(pet_id: str):
    pet = Collections["pets"].find_one({"id": pet_id})
    if not pet:
        return

    defaults = {
        "alive": True,
        "last_fed": int(time.time()),
        "level": 1,
        "exp": 0,
        "benefits": {"token_bonus": 1},
        "base_price": 100,
        "hunger": 100,
        "happiness": 100,
        "last_play_time": int(time.time()),  # Initialize last_play_time
        "last_update_time": int(time.time()),  # Track the last update time
        "personality": random.choice(["Playful", "Lazy", "Adventurous", "Hungry"]),
    }
    updates = {}
    for key, value in defaults.items():
        if key not in pet:
            updates[key] = value
    if updates:
        Collections["pets"].update_one({"id": pet_id}, {"$set": updates})

    pet = Collections["pets"].find_one({"id": pet_id})

    last_fed = pet["last_fed"]
    last_play_time = pet.get("last_play_time", last_fed)
    last_update_time = pet.get("last_update_time", last_fed)
    now = int(time.time())

    # Only update hunger and happiness if enough time has passed
    update_interval = 60 * 60  # 1 hour in seconds
    if now - last_update_time >= update_interval:
        # Health status and death check
        if pet["alive"]:
            seconds_unfed = now - last_fed
            seconds_unplayed = now - last_play_time

            hunger_rate = 1
            happiness_rate = 1

            if pet["personality"] == "Lazy":
                happiness_rate = 0.5

            if pet["personality"] == "Hungry":
                hunger_rate = 2

            new_hunger = max(0, pet["hunger"] - (seconds_unfed * (hunger_rate / 3600)))
            new_happiness = max(
                0, pet["happiness"] - (seconds_unplayed * (happiness_rate / 3600))
            )

            if new_hunger <= 0 or new_happiness <= 0:
                Collections["pets"].update_one(
                    {"id": pet_id},
                    {"$set": {"hunger": 0, "happiness": 0, "alive": False}},
                )
                send_discord_notification(
                    "Pet Died",
                    f"User {pet['owner']}'s pet {pet['name']} died due to neglect.",
                    0xFF0000,
                )
            else:
                Collections["pets"].update_one(
                    {"id": pet_id},
                    {
                        "$set": {
                            "hunger": new_hunger,
                            "happiness": new_happiness,
                            "last_update_time": now,
                        }
                    },
                )

        # Update benefits based on level (only if alive)
        if pet["alive"]:
            pet["benefits"]["token_bonus"] = pet["level"]  # +1 token per level
            Collections["pets"].update_one(
                {"id": pet_id}, {"$set": {"benefits": pet["benefits"]}}
            )


def level_up_pet(pet_id: str, exp_gain: int):
    pet = Collections["pets"].find_one({"id": pet_id})
    if not pet or not pet["alive"]:
        return

    new_exp = pet["exp"] + exp_gain
    next_level_exp = exp_for_level(pet["level"] + 1)
    if new_exp >= next_level_exp:
        Collections["pets"].update_one(
            {"id": pet_id},
            {"$set": {"level": pet["level"] + 1, "exp": new_exp - next_level_exp}},
        )
        send_discord_notification(
            "Pet Leveled Up",
            f"User {pet['owner']}'s pet {pet['name']} reached level {pet['level'] + 1}!",
            0x00FF00,
        )
    else:
        Collections["pets"].update_one({"id": pet_id}, {"$set": {"exp": new_exp}})


def update_account(username: str) -> Optional[Tuple[dict, int]]:
    user = Collections["users"].find_one({"username": username})
    if not user:
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404

    defaults = {
        "banned_until": None,
        "banned_reason": None,
        "banned": False,
        "history": [],
        "exp": 0,
        "level": 1,
        "frozen": False,
        "muted": False,
        "muted_until": None,
        "inventory_visibility": "private",
        "2fa_enabled": False,
        "pets": [],
        "override_plan": None,
        "override_plan_expires": None,
        "redeemed_creator_code": False,
        "creator_code": None,
        "gems": 0,
        "cosmetics": [],
        "equipped_nameplate": None,
        "equipped_messageplate": None,
        "subscriptions": [],
    }
    updates = {k: v for k, v in defaults.items() if k not in user}
    if updates:
        Collections["users"].update_one({"username": username}, {"$set": updates})

    current_time = time.time()
    if (
        user.get("banned_until")
        and user["banned_until"] < current_time
        and user["banned_until"] != 0
    ):
        Collections["users"].update_one(
            {"username": username},
            {"$set": {"banned_until": None, "banned_reason": None, "banned": False}},
        )
    if (
        user.get("muted_until")
        and user["muted_until"] < current_time
        and user["muted_until"] != 0
    ):
        Collections["users"].update_one(
            {"username": username}, {"$set": {"muted": False, "muted_until": None}}
        )

    for item_id in user["items"]:
        update_item(item_id)

    pet_limit = 2
    if has_pro(username):
        pet_limit = 4
    if has_proplus(username):
        pet_limit = 8

    if len(user["pets"]) > pet_limit:
        refund = (len(user["pets"]) - 1) * 100
        Collections["users"].update_one(
            {"username": username},
            {"$set": {"pets": [user["pets"][0]]}, "$inc": {"tokens": refund}},
        )
    for pet_id in user["pets"]:
        update_pet(pet_id)

    if has_pro(username) and "gold" not in user["cosmetics"]:
        Collections["users"].update_one(
            {"username": username},
            {"$push": {"cosmetics": "gold"}},
        )

    remove_companies()


# Item and Pet Generation
def generate_item(owner: str) -> dict:
    def weighted_choice(items: dict, special_case: bool = False):
        choices, weights = zip(*items.items())
        if special_case:
            weights = [1 / items[c]["rarity"] for c in choices]
        return random.choices(choices, weights=weights, k=1)[0]

    noun = weighted_choice(NOUNS, special_case=True)
    name = {
        "adjective": weighted_choice(ADJECTIVES),
        "material": weighted_choice(MATERIALS),
        "noun": noun,
        "suffix": weighted_choice(SUFFIXES),
        "number": random.randint(1, 9999),
        "icon": NOUNS[noun]["icon"],
    }
    meta_id = sha256(
        f"{name['adjective']}{name['material']}{name['noun']}{name['suffix']}".encode()
    ).hexdigest()

    lore = generate_lore(name)

    meta = Collections["item_meta"].find_one({"id": meta_id})
    if not meta:
        rarity = round(random.uniform(0.05, 100), 2)
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
            "price_history": [],
            "lore": lore,
        }
        Collections["item_meta"].insert_one(meta)

    return {
        "id": str(uuid4()),
        "meta_id": meta_id,
        "item_secret": str(uuid4()),
        "rarity": meta["rarity"],
        "level": meta["level"],
        "name": name,
        "history": [],
        "for_sale": False,
        "price": 0,
        "owner": owner,
        "created_at": int(time.time()),
        "lore": meta["lore"],
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
        "benefits": {"token_bonus": 1},
        "alive": True,
        "base_price": base_price,
        "hunger": 0,
        "happiness": 100,
        "personality": random.choice(["Playful", "Lazy", "Adventurous", "Hungry"]),
    }


# Experience System
def exp_for_level(level: int) -> int:
    return int(25 * (1.2 ** (level - 1)))


def add_exp(username: str, exp: int):
    user = Collections["users"].find_one({"username": username})
    if not user:
        return

    # Apply EXP multiplier based on plan
    plan = get_plan(username)
    if plan == "pro":
        exp = int(exp * 1.5)
    elif plan == "proplus":
        exp = int(exp * 2.0)

    new_exp = user["exp"] + exp
    Collections["users"].update_one({"username": username}, {"$set": {"exp": new_exp}})
    if new_exp >= exp_for_level(user["level"] + 1):
        Collections["users"].update_one(
            {"username": username}, {"$set": {"level": user["level"] + 1}}
        )


def set_exp(username: str, exp: int):
    user = Collections["users"].find_one({"username": username})
    if not user:
        return
    Collections["users"].update_one({"username": username}, {"$set": {"exp": exp}})
    if exp >= exp_for_level(user["level"] + 1):
        Collections["users"].update_one(
            {"username": username}, {"$set": {"level": user["level"] + 1}}
        )


def set_level(username: str, level: int):
    user = Collections["users"].find_one({"username": username})
    if not user:
        return
    level_exp = exp_for_level(level)
    Collections["users"].update_one(
        {"username": username}, {"$set": {"level": level, "exp": level_exp}}
    )


# Core Handlers
def register(username: str, password: str, ip: str) -> Tuple[dict, int]:
    if is_ip_blocked(ip):
        return jsonify({"error": "IP blocked", "code": "ip-blocked"}), 403

    if not username or not password:
        return (
            jsonify({"error": "Missing credentials", "code": "missing-credentials"}),
            400,
        )

    current_time = time.time()
    recent_attempts = Collections["account_creation_attempts"].count_documents(
        {
            "ip": ip,
            "timestamp": {
                "$gt": current_time - AUTOMOD_CONFIG["ACCOUNT_CREATION_TIME_WINDOW"]
            },
        }
    )

    if recent_attempts >= AUTOMOD_CONFIG["ACCOUNT_CREATION_THRESHOLD"]:
        block_ip(
            ip,
            str(AUTOMOD_CONFIG["ACCOUNT_CREATION_BLOCK_DURATION"] + "s"),
            "Account creation spam",
            subnet=True,
        )
        Collections["users"].update_many(
            {"creation_ip": ip},
            {
                "$set": {
                    "banned": True,
                    "banned_until": 0,
                    "banned_reason": "AutoMod: Account spam",
                }
            },
        )
        Collections["messages"].insert_one(
            {
                "id": str(uuid4()),
                "room": "global",
                "username": "AutoMod",
                "message": f"""
            <p><span style="color: #FF5555">[WARNING]</span> Detected <b>{recent_attempts + 1}x</b> Account Creation Spam</p>
            <p>IP: <b>{ip}</b> has been blocked for <b>{AUTOMOD_CONFIG['ACCOUNT_CREATION_BLOCK_DURATION']} seconds</b></p>
            """,
                "timestamp": current_time,
                "type": "system",
            }
        )
        return jsonify({"error": "Account spam detected", "code": "account-spam"}), 429

    username = profanity.censor(username.strip(), censor_char="-")
    if not re.match(r"^[a-zA-Z0-9_-]{3,20}$", username):
        return jsonify({"error": "Invalid username"}), 400

    try:
        user_data = {
            "created_at": int(time.time()),
            "username": username,
            "password_hash": generate_password_hash(password),
            "type": "user",
            "tokens": 100,
            "last_item_time": 0,
            "last_mine_time": 0,
            "items": [],
            "token": None,
            "banned_until": None,
            "banned_reason": None,
            "banned": False,
            "muted": False,
            "muted_until": None,
            "history": [],
            "exp": 0,
            "level": 1,
            "2fa_enabled": False,
            "inventory_visibility": "private",
            "pets": [],
            "creation_ip": ip,
            "override_plan": None,
            "override_plan_expires": None,
            "redeemed_creator_code": False,
            "creator_code": None,
            "gems": 0,
            "cosmetics": [],
            "equipped_messageplate": None,
            "equipped_nameplate": None,
            "subscriptions": [],
        }
        Collections["users"].insert_one(user_data)
        Collections["account_creation_attempts"].insert_one(
            {"ip": ip, "timestamp": current_time}
        )
        send_discord_notification(
            "New user registered", f"Username: {username}\nIP: {ip}"
        )
        plugin_manager.trigger_hooks(
            "on_user_register", {"username": username, "ip": ip}, {"db": db}
        )
        return jsonify({"success": True}), 201
    except DuplicateKeyError:
        return jsonify({"error": "Username exists", "code": "username-exists"}), 400


def login(
    username: str,
    password: str,
    ip: str,
    code: Optional[str] = None,
    token: Optional[str] = None,
) -> Tuple[dict, int]:
    if is_ip_blocked(ip):
        return jsonify({"error": "IP blocked", "code": "ip-blocked"}), 403

    recent_fails = Collections["failed_logins"].count_documents(
        {
            "ip": ip,
            "timestamp": {"$gt": time.time() - AUTOMOD_CONFIG["FAILED_LOGIN_WINDOW"]},
        }
    )
    if recent_fails >= AUTOMOD_CONFIG["FAILED_LOGIN_THRESHOLD"]:
        block_ip(
            ip,
            str(AUTOMOD_CONFIG["FAILED_LOGIN_WINDOW"]) + "s",
            "Too many failed logins",
            subnet=True,
        )
        return (
            jsonify({"error": "Too many failed attempts", "code": "login-locked"}),
            429,
        )

    user = Collections["users"].find_one({"username": username})
    if not user or not check_password_hash(user["password_hash"], password):
        Collections["failed_logins"].insert_one({"ip": ip, "timestamp": time.time()})
        return (
            jsonify({"error": "Invalid credentials", "code": "invalid-credentials"}),
            401,
        )

    if user.get("2fa_enabled", False):
        if not code and not token:
            return jsonify({"error": "2FA required", "code": "2fa-required"}), 401
        if code:
            if user["2fa_code"] != code:
                return (
                    jsonify({"error": "Invalid 2FA code", "code": "invalid-2fa-code"}),
                    401,
                )
        else:
            totp = pyotp.TOTP(user["2fa_secret"])
            if not totp.verify(token):
                return (
                    jsonify(
                        {"error": "Invalid 2FA token", "code": "invalid-2fa-token"}
                    ),
                    401,
                )

    token = str(uuid4())
    Collections["users"].update_one({"username": username}, {"$set": {"token": token}})
    send_discord_notification("User logged in", f"Username: {username}")
    plugin_manager.trigger_hooks(
        "on_user_login", {"username": username, "ip": ip}, {"db": db}
    )
    return jsonify({"success": True, "token": token})


def get_users() -> Tuple[dict, int]:
    users = Collections["users"].find({}, {"_id": 0, "username": 1})
    return jsonify({"usernames": [user["username"] for user in users]})


def get_user(username: str) -> Tuple[dict, int]:
    user = Collections["users"].find_one({"username": username}, {"_id": 0})
    return jsonify(user)


def parse_command(username: str, command: str, room_name: str) -> str:
    user = Collections["users"].find_one({"username": username})
    is_admin = user.get("type") == "admin"
    is_mod = user.get("type") in ["admin", "mod"]

    parts = command[1:].split(" ")
    cmd, *args = parts

    if cmd == "clear_chat" and is_admin:
        Collections["messages"].delete_many({"room": room_name})
        return f"Cleared chat in <b>{room_name}</b>"
    elif cmd == "clear_user" and len(args) == 1 and is_admin:
        Collections["messages"].delete_many({"room": room_name, "username": args[0]})
        return f"Deleted messages from <b>{args[0]}</b> in <b>{room_name}</b>"
    elif cmd == "delete_many" and len(args) == 1 and is_admin:
        try:
            amount = int(args[0])
            messages = (
                Collections["messages"]
                .find({"room": room_name})
                .sort("timestamp", DESCENDING)
                .limit(amount)
            )
            ids = [doc["_id"] for doc in messages]
            Collections["messages"].delete_many({"_id": {"$in": ids}})
            return f"Deleted <b>{amount}</b> messages from <b>{room_name}</b>"
        except ValueError:
            return "Invalid amount specified"
    elif cmd == "ban" and len(args) >= 3 and is_admin:
        target, duration, *reason = args
        ban_user(target, duration, " ".join(reason))
        return (
            f"Banned <b>{target}</b> for <b>{' '.join(reason)}</b> (<b>{duration}</b>)"
        )
    elif cmd == "mute" and len(args) == 2 and is_mod:
        mute_user(args[0], args[1])
        return f"Muted <b>{args[0]}</b> for <b>{args[1]}</b>"
    elif cmd == "unban" and len(args) == 1 and is_admin:
        Collections["users"].update_one(
            {"username": args[0]}, {"$set": {"banned": False}}
        )
        return f"Unbanned <b>{args[0]}</b>"
    elif cmd == "unmute" and len(args) == 1 and is_mod:
        unmute_user(args[0])
        return f"Unmuted <b>{args[0]}</b>"
    elif cmd == "sudo" and len(args) >= 2 and is_admin:
        sudo_user = args[0]
        message = " ".join(args[1:])
        user_data = Collections["users"].find_one({"username": sudo_user})
        if not user_data:
            return f"User <b>{sudo_user}</b> not found"
        badges = []
        if user_data["type"] == "mod":
            badges.append("🛡️")
        elif user_data["type"] == "admin":
            badges.append("🛠️")
        elif user_data["type"] == "media":
            badges.append("🎥")

        if has_proplus(sudo_user):
            badges.append("🌟")
        elif has_pro(sudo_user):
            badges.append("⭐")
        Collections["messages"].insert_one(
            {
                "id": str(uuid4()),
                "room": room_name,
                "username": sudo_user,
                "message": message,
                "timestamp": time.time(),
                "badges": badges,
                "type": user_data["type"],
                "messageplate": user_data.get("equipped_messageplate", None),
                "nameplate": user_data.get("equipped_nameplate", None),
            }
        )
        return None
    elif cmd == "list_banned" and is_mod:
        banned = Collections["users"].find({"banned": True})
        banned_list = [
            f"<b>{u['username']}</b> - {u.get('banned_reason', 'No reason')}"
            for u in banned
        ]
        return (
            "Banned users:\n" + "\n".join(banned_list)
            if banned_list
            else "Nobody is banned."
        )
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
    elif cmd == "msg" and len(args) >= 2:
        recipient = args[0]
        message = " ".join(args[1:])
        Collections["messages"].insert_one(
            {
                "id": str(uuid4()),
                "room": room_name,
                "username": f"{username} -> {recipient}",
                "message": message,
                "timestamp": time.time(),
                "badges": ["📩"],
                "visibility": [username, recipient],
                "type": "msg",
            }
        )
    else:
        return "Invalid command"


def send_message(
    room_name: str, message_content: str, username: str, ip: str
) -> Tuple[dict, int]:
    user = Collections["users"].find_one({"username": username})
    if user["muted"]:
        return jsonify({"error": "You are muted", "code": "user-muted"}), 400

    if not room_name or not message_content:
        return (
            jsonify({"error": "Missing room or message", "code": "missing-parameters"}),
            400,
        )

    if room_name == "exclusive":
        if not (has_pro(username) or request.user_type in ["admin"]):
            return jsonify({"error": "Access denied", "code": "access-denied"}), 403
    if room_name == "staff":
        if request.user_type not in ["mod", "admin"]:
            return jsonify({"error": "Access denied", "code": "access-denied"}), 403
    if room_name not in ["global", "exclusive", "staff"]:
        return jsonify({"error": "Invalid room", "code": "invalid-room"}), 400

    if not re.match(r"^[a-zA-Z0-9_-]{1,50}$", room_name):
        return jsonify({"error": "Invalid room name", "code": "invalid-room"}), 400

    current_time = time.time()
    message_count = Collections["message_attempts"].count_documents(
        {
            "username": username,
            "timestamp": {
                "$gt": current_time - AUTOMOD_CONFIG["MESSAGE_SPAM_TIME_WINDOW"]
            },
        }
    )

    if message_count >= AUTOMOD_CONFIG["MESSAGE_SPAM_THRESHOLD"]:
        is_new = (current_time - user["created_at"]) < AUTOMOD_CONFIG["MIN_ACCOUNT_AGE"]
        mute_duration = (
            AUTOMOD_CONFIG["NEW_USER_MESSAGE_SPAM_MUTE_DURATION"]
            if is_new
            else AUTOMOD_CONFIG["MESSAGE_SPAM_MUTE_DURATION"]
        )
        mute_user(username, mute_duration, notify=False)
        deleted = (
            Collections["messages"]
            .delete_many(
                {
                    "username": username,
                    "timestamp": {
                        "$gt": current_time - AUTOMOD_CONFIG["MESSAGE_SPAM_TIME_WINDOW"]
                    },
                }
            )
            .deleted_count
        )
        Collections["messages"].insert_one(
            {
                "id": str(uuid4()),
                "room": room_name,
                "username": "AutoMod",
                "message": f"""
            <p><span style="color: #FF5555">[WARNING]</span> Detected <b>{deleted}x</b> Message Spam</p>
            <p>User: <b>{username}</b> has been muted for <b>{mute_duration}</b></p>
            """,
                "timestamp": current_time,
                "badges": ["⚙️"],
                "type": "system",
            }
        )
        send_discord_notification(
            "AutoMod Action",
            f"Muted {username} for spamming. Deleted {deleted} messages.",
            0xFF0000,
        )
        return jsonify({"error": "Message spam detected", "code": "message-spam"}), 429

    Collections["message_attempts"].insert_one(
        {"username": username, "ip": ip, "timestamp": current_time}
    )
    sanitized_message = profanity.censor(html.escape(message_content.strip()))
    if not sanitized_message:
        return jsonify({"error": "Message empty", "code": "empty-message"}), 400
    if len(sanitized_message) > 200:
        return jsonify({"error": "Message too long", "code": "message-too-long"}), 400

    if sanitized_message.startswith("/"):
        system_message = parse_command(username, sanitized_message, room_name)
        if system_message:
            Collections["messages"].insert_one(
                {
                    "id": str(uuid4()),
                    "room": room_name,
                    "username": "Command Handler",
                    "message": system_message,
                    "timestamp": time.time(),
                    "badges": ["⚙️"],
                    "type": "system",
                }
            )
    else:
        badges = []
        if user["type"] == "mod":
            badges.append("🛡️")
        elif user["type"] == "admin":
            badges.append("🛠️")
        elif user["type"] == "media":
            badges.append("🎥")

        if has_proplus(username):
            badges.append("🌟")
        elif has_pro(username):
            badges.append("⭐")

        Collections["messages"].insert_one(
            {
                "id": str(uuid4()),
                "room": room_name,
                "username": username,
                "message": sanitized_message,
                "timestamp": time.time(),
                "nameplate": user.get("equipped_nameplate", None),
                "messageplate": user.get("equipped_messageplate", None),
                "badges": badges,
                "type": user["type"],
            }
        )
    return jsonify({"success": True})


# Admin/Mod Functions
def reset_cooldowns(username: str) -> Tuple[dict, int]:
    Collections["users"].update_one(
        {"username": username}, {"$set": {"last_item_time": 0, "last_mine_time": 0}}
    )
    send_discord_notification(
        "Cooldowns Reset",
        f"Admin {request.username} reset cooldowns for {username}",
        0xFFA500,
    )
    return jsonify({"success": True})


def edit_tokens(username: str, tokens: float) -> Tuple[dict, int]:
    try:
        tokens = float(tokens)
    except ValueError:
        return (
            jsonify({"error": "Invalid tokens value", "code": "invalid-tokens-value"}),
            400,
        )

    if not Collections["users"].find_one({"username": username}):
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404

    Collections["users"].update_one(
        {"username": username}, {"$set": {"tokens": tokens}}
    )
    send_discord_notification(
        "Tokens Edited",
        f"Admin {request.username} set {username}'s tokens to {tokens}",
        0xFFA500,
    )
    return jsonify({"success": True})


def edit_exp(username: str, exp: float) -> Tuple[dict, int]:
    try:
        exp = float(exp)
        if exp < 0:
            return (
                jsonify(
                    {"error": "Exp cannot be negative", "code": "cannot-be-negative"}
                ),
                400,
            )
    except ValueError:
        return jsonify({"error": "Invalid exp value", "code": "invalid-value"}), 400

    if not Collections["users"].find_one({"username": username}):
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404

    set_exp(username, exp)
    send_discord_notification(
        "Experience Edited",
        f"Admin {request.username} set {username}'s exp to {exp}",
        0xFFA500,
    )
    return jsonify({"success": True})


def edit_level(username: str, level: int) -> Tuple[dict, int]:
    try:
        level = int(level)
        if level < 1:
            return (
                jsonify(
                    {
                        "error": "Level cannot be less than 1",
                        "code": "cannot-be-negative",
                    }
                ),
                400,
            )
    except ValueError:
        return jsonify({"error": "Invalid level value", "code": "invalid-value"}), 400

    if not Collections["users"].find_one({"username": username}):
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404

    set_level(username, level)
    send_discord_notification(
        "Level Edited",
        f"Admin {request.username} set {username}'s level to {level}",
        0xFFA500,
    )
    return jsonify({"success": True})


def add_admin(username: str) -> Tuple[dict, int]:
    if not Collections["users"].find_one({"username": username}):
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404

    Collections["users"].update_one({"username": username}, {"$set": {"type": "admin"}})
    send_discord_notification(
        "Admin Added", f"Admin {request.username} added {username} as admin", 0xFFA500
    )
    return jsonify({"success": True})


def remove_admin(username: str) -> Tuple[dict, int]:
    if not Collections["users"].find_one({"username": username}):
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404

    Collections["users"].update_one({"username": username}, {"$set": {"type": "user"}})
    send_discord_notification(
        "Admin Removed",
        f"Admin {request.username} removed {username} as admin",
        0xFFA500,
    )
    return jsonify({"success": True})


def add_mod(username: str) -> Tuple[dict, int]:
    if not Collections["users"].find_one({"username": username}):
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404

    Collections["users"].update_one({"username": username}, {"$set": {"type": "mod"}})
    send_discord_notification(
        "Mod Added", f"Admin {request.username} added {username} as mod", 0xFFA500
    )
    return jsonify({"success": True})


def remove_mod(username: str) -> Tuple[dict, int]:
    if not Collections["users"].find_one({"username": username}):
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404

    Collections["users"].update_one({"username": username}, {"$set": {"type": "user"}})
    send_discord_notification(
        "Mod Removed", f"Admin {request.username} removed {username} as mod", 0xFFA500
    )
    return jsonify({"success": True})


def edit_item(
    item_id: str, new_name: str, new_icon: str, new_rarity: str
) -> Tuple[dict, int]:
    item = Collections["items"].find_one({"id": item_id})
    if not item:
        return jsonify({"error": "Item not found", "code": "item-not-found"}), 404

    updates = {}
    if new_name:
        parts = split_name(new_name)
        updates.update(
            {
                "name.adjective": html.escape(parts["adjective"].strip()),
                "name.material": html.escape(parts["material"].strip()),
                "name.noun": html.escape(parts["noun"].strip()),
                "name.suffix": html.escape(parts["suffix"].strip()),
                "name.number": html.escape(parts["number"].strip()),
            }
        )
    if new_icon:
        updates["name.icon"] = html.escape(new_icon.strip())
    if new_rarity:
        updates["rarity"] = float(new_rarity)
        updates["level"] = get_level(float(new_rarity))

    if updates:
        Collections["items"].update_one({"id": item_id}, {"$set": updates})
        updated_item = Collections["items"].find_one({"id": item_id}, {"_id": 0})
        item_name = " ".join(
            [
                updated_item["name"]["adjective"],
                updated_item["name"]["material"],
                updated_item["name"]["noun"],
                updated_item["name"]["suffix"],
                f"#{updated_item['name']['number']}",
            ]
        ).strip()
        updates_str = ", ".join([f"{k}: {v}" for k, v in updates.items()])
        send_discord_notification(
            "Item Edited",
            f"Admin {request.username} edited {item_name} (ID: {item_id}). Changes: {updates_str}",
            0xFFA500,
        )
    return jsonify({"success": True})


def delete_item(item_id: str) -> Tuple[dict, int]:
    item = Collections["items"].find_one({"id": item_id})
    if not item:
        return jsonify({"error": "Item not found", "code": "item-not-found"}), 404

    Collections["users"].update_one(
        {"username": item["owner"]}, {"$pull": {"items": item_id}}
    )
    Collections["items"].delete_one({"id": item_id})
    send_discord_notification(
        "Item Deleted", f"Admin {request.username} deleted item {item_id}", 0xFF0000
    )
    return jsonify({"success": True})


def ban_user(username: str, length: str, reason: str) -> Tuple[dict, int]:
    user = Collections["users"].find_one({"username": username})
    if not user:
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404
    if user.get("type") == "admin":
        return jsonify({"error": "Cannot ban admin", "code": "cannot-ban-admin"}), 403

    for sub in user.get("subscriptions", []):
        if sub["status"] == "active":
            try:
                subscription = stripe.Subscription.retrieve(sub["subscription_id"])
                subscription.delete()
            except stripe.error.StripeError as e:
                app.logger.error(f"Error cancelling subscription: {str(e)}")

    end_time = parse_time(length)
    Collections["users"].update_one(
        {"username": username},
        {"$set": {"banned_until": end_time, "banned_reason": reason, "banned": True}},
    )
    send_discord_notification(
        "User Banned",
        f"Admin {request.username} banned {username} for {length}. Reason: {reason}",
        0xFF0000,
    )
    return jsonify({"success": True})


def unban_user(username: str) -> Tuple[dict, int]:
    if not Collections["users"].find_one({"username": username}):
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404

    Collections["users"].update_one(
        {"username": username},
        {"$set": {"banned_until": None, "banned_reason": None, "banned": False}},
    )
    send_discord_notification(
        "User Unbanned", f"Admin {request.username} unbanned {username}", 0xFFA500
    )
    return jsonify({"success": True})


def mute_user(username: str, length: str, notify: bool = True) -> Tuple[dict, int]:
    user_to_mute = Collections["users"].find_one({"username": username})
    if not user_to_mute:
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404

    if user_to_mute.get("type") in ["mod", "admin"] and request.user_type == "mod":
        return (
            jsonify(
                {"error": "Cannot mute other mods or admins", "code": "unauthorized"}
            ),
            403,
        )

    end_time = parse_time(length)
    Collections["users"].update_one(
        {"username": username}, {"$set": {"muted_until": end_time, "muted": True}}
    )
    if notify:
        send_discord_notification(
            "User Muted",
            f"Admin/Mod {request.username} muted {username} for {length}",
            0xFFA500,
        )
    return jsonify({"success": True})


def unmute_user(username: str, notify: bool = True) -> Tuple[dict, int]:
    user_to_unmute = Collections["users"].find_one({"username": username})
    if not user_to_unmute:
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404

    if user_to_unmute.get("type") in ["mod", "admin"] and request.user_type == "mod":
        return (
            jsonify(
                {"error": "Cannot unmute other mods or admins", "code": "unauthorized"}
            ),
            403,
        )

    Collections["users"].update_one(
        {"username": username}, {"$set": {"muted": False, "muted_until": None}}
    )
    if notify:
        send_discord_notification(
            "User Unmuted", f"Admin/Mod {request.username} unmuted {username}", 0xFFA500
        )
    return jsonify({"success": True})


def fine_user(username: str, amount: int) -> Tuple[dict, int]:
    if not Collections["users"].find_one({"username": username}):
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404

    Collections["users"].update_one(
        {"username": username}, {"$inc": {"tokens": -amount}}
    )
    send_discord_notification(
        "User Fined",
        f"Admin {request.username} fined {username} {amount} tokens",
        0xFFA500,
    )
    return jsonify({"success": True})


def delete_message(message_id: str) -> Tuple[dict, int]:
    if not message_id:
        return (
            jsonify({"error": "Missing message_id", "code": "missing-parameters"}),
            400,
        )

    Collections["messages"].delete_one({"id": message_id})
    send_discord_notification(
        "Message Deleted",
        f"Mod/Admin {request.username} deleted message {message_id}",
        0xFF0000,
    )
    return jsonify({"success": True})


def get_banned_ips() -> Tuple[dict, int]:
    current_time = time.time()
    banned_ips = Collections["blocked_ips"].find(
        {"blocked_until": {"$gte": current_time}}, {"_id": 0}
    )
    return jsonify({"banned_ips": list(banned_ips)})


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
    return login(
        data.get("username"),
        data.get("password"),
        request.remote_addr,
        data.get("code"),
        data.get("token"),
    )


@app.route("/api/setup_2fa", methods=["POST"])
@requires_unbanned
def setup_2fa_endpoint():
    user = Collections["users"].find_one({"username": request.username})
    if user.get("2fa_enabled", False):
        return (
            jsonify({"error": "2FA already enabled", "code": "2fa-already-enabled"}),
            400,
        )

    secret = user.get("2fa_secret") or pyotp.random_base32(32)
    code = user.get("2fa_code") or str(uuid4())
    Collections["users"].update_one(
        {"username": request.username},
        {"$set": {"2fa_secret": secret, "2fa_code": code}},
    )
    totp = pyotp.TOTP(secret)
    uri = totp.provisioning_uri(
        name=request.username,
        issuer_name="Economix",
        image="https://economix.lol/brand/logo.png",
    )
    send_discord_notification("2FA enabled", f"Username: {request.username}")
    img = qrcode.make(uri)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/api/verify_2fa", methods=["POST"])
@requires_unbanned
def verify_2fa_endpoint():
    data = request.get_json()
    user = Collections["users"].find_one({"username": request.username})
    if "2fa_secret" not in user:
        return jsonify({"error": "2FA not setup", "code": "2fa-not-setup"}), 400
    totp = pyotp.TOTP(user["2fa_secret"])
    if not totp.verify(data.get("code")):
        return jsonify({"error": "Invalid 2FA token", "code": "invalid-2fa-token"}), 401
    Collections["users"].update_one(
        {"username": request.username}, {"$set": {"2fa_enabled": True}}
    )
    return jsonify({"success": True, "backup_code": user["2fa_code"]})


@app.route("/api/disable_2fa", methods=["POST"])
@requires_unbanned
def disable_2fa_endpoint():
    Collections["users"].update_one(
        {"username": request.username},
        {"$set": {"2fa_enabled": False, "2fa_secret": None, "2fa_code": None}},
    )
    send_discord_notification("2FA disabled", f"Username: {request.username}")
    return jsonify({"success": True})


@app.route("/api/account", methods=["GET"])
def account_endpoint():
    update_account(request.username)
    user = Collections["users"].find_one({"username": request.username})
    items = list(Collections["items"].find({"id": {"$in": user["items"]}}, {"_id": 0}))
    pets = list(Collections["pets"].find({"id": {"$in": user["pets"]}}, {"_id": 0}))
    return jsonify(
        {
            "username": user["username"],
            "type": user.get("type", "user"),
            "tokens": user["tokens"],
            "items": items,
            "last_item_time": user["last_item_time"],
            "last_mine_time": user["last_mine_time"],
            "banned_until": user.get("banned_until"),
            "banned_reason": user.get("banned_reason"),
            "banned": user.get("banned"),
            "muted": user.get("muted"),
            "muted_until": user.get("muted_until"),
            "exp": user.get("exp"),
            "level": user.get("level"),
            "history": user.get("history"),
            "2fa_enabled": user.get("2fa_enabled"),
            "pets": pets,
            "override_plan": user.get("override_plan"),
            "override_plan_expires": user.get("override_plan_expires"),
            "redeemed_creator_code": user.get("redeemed_creator_code"),
            "creator_code": user.get("creator_code"),
            "plan": get_plan(request.username),
            "gems": user.get("gems", 0),
            "cosmetics": user.get("cosmetics", []),
            "equipped_nameplate": user.get("equipped_nameplate"),
            "equipped_messageplate": user.get("equipped_messageplate"),
        }
    )


@app.route("/api/delete_account", methods=["POST"])
@requires_unbanned
def delete_account_endpoint():
    user = Collections["users"].find_one({"username": request.username})
    if not user:
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404

    # Check for suspicious token transfers before deletion
    recent_transfers = Collections["messages"].count_documents(
        {
            "room": "logs",
            "username": request.username,
            "type": "system",
            "timestamp": {
                "$gt": time.time() - AUTOMOD_CONFIG["EXPLOIT_DETECTION_TIME_WINDOW"]
            },
            "message": {"$regex": "sent .* tokens"},
        }
    )
    if recent_transfers > 0:
        send_discord_notification(
            "Suspicious Account Deletion",
            f"User {request.username} attempted to delete their account after transferring tokens.",
            0xFF0000,
        )
        return (
            jsonify(
                {
                    "error": "Suspicious activity detected. Account deletion blocked.",
                    "code": "suspicious-deletion",
                }
            ),
            403,
        )

    for sub in user.get("subscriptions", []):
        if sub["status"] == "active":
            try:
                subscription = stripe.Subscription.retrieve(sub["subscription_id"])
                subscription.delete()
            except stripe.error.StripeError as e:
                app.logger.error(f"Error cancelling subscription: {str(e)}")

    Collections["items"].delete_many({"owner": request.username})
    Collections["users"].delete_one({"username": request.username})
    send_discord_notification("User deleted", f"Username: {request.username}")
    plugin_manager.trigger_hooks(
        "on_account_delete", {"username": request.username}, {"db": db}
    )
    return jsonify({"success": True})


@app.route("/api/create_item", methods=["POST"])
@requires_unbanned
def create_item_endpoint():
    now = time.time()
    user = Collections["users"].find_one({"username": request.username}, {"_id": 0})
    if not user:
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404
    if now - user["last_item_time"] < ITEM_CREATE_COOLDOWN:
        return (
            jsonify(
                {
                    "error": "Cooldown active",
                    "remaining": ITEM_CREATE_COOLDOWN - (now - user["last_item_time"]),
                    "code": "cooldown-active",
                }
            ),
            429,
        )
    if user["tokens"] < 10:
        return jsonify({"error": "Not enough tokens", "code": "not-enough-tokens"}), 402

    new_item = generate_item(request.username)
    Collections["items"].insert_one(new_item)
    Collections["users"].update_one(
        {"username": request.username},
        {
            "$push": {"items": new_item["id"]},
            "$set": {"last_item_time": now},
            "$inc": {"tokens": -10},
        },
    )
    Collections["users"].update_one(
        {"username": request.username},
        {
            "$push": {
                "history": {
                    "item_id": new_item["id"],
                    "action": "create",
                    "timestamp": now,
                }
            }
        },
    )
    add_exp(request.username, 10)

    item_name = " ".join(
        [
            new_item["name"]["adjective"],
            new_item["name"]["material"],
            new_item["name"]["noun"],
            new_item["name"]["suffix"],
            f"#{new_item['name']['number']}",
        ]
    ).strip()
    send_discord_notification(
        "New Item Created",
        f"User {request.username} created: {item_name} (Rarity: {new_item['rarity']})",
    )
    plugin_manager.trigger_hooks(
        "on_item_create", {"username": request.username, "item": new_item}, {"db": db}
    )
    return jsonify(
        {k: v for k, v in new_item.items() if k not in ["_id", "item_secret"]}
    )


@app.route("/api/buy_pet", methods=["POST"])
@requires_unbanned
def buy_pet_endpoint():
    user = Collections["users"].find_one({"username": request.username})
    if not user:
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404

    pet_limit = 2
    if has_pro(request.username):
        pet_limit = 4
    if has_proplus(request.username):
        pet_limit = 8

    if len(user.get("pets", [])) >= pet_limit:
        for pet_id in user["pets"]:
            current_pet = Collections["pets"].find_one({"id": pet_id})
            if current_pet["alive"]:
                return (
                    jsonify(
                        {
                            "error": "Already have maximum pets",
                            "code": "user-already-has-pet",
                        }
                    ),
                    400,
                )
            # Remove dead pet
            Collections["users"].update_one(
                {"username": request.username}, {"$pull": {"pets": current_pet["id"]}}
            )
            price = 200  # Double price for repurchase
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
    return jsonify(
        {"success": True, "pet": {k: v for k, v in pet.items() if k not in ["_id"]}}
    )


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
    if pet["personality"] == "Hungry":
        pet["happiness"] += 15  # Extra happiness for Hungry pets
    else:
        pet["happiness"] += 10
    Collections["pets"].update_one(
        {"id": pet_id},
        {
            "$inc": {
                "happiness": min(10, 100 - pet["happiness"]),
                "hunger": min(10, 100 - pet["hunger"]),
            }
        },
    )
    level_up_pet(pet_id, 3)  # Gain 3 exp per feeding
    update_pet(pet_id)
    send_discord_notification(
        "Pet Fed", f"User {request.username} fed pet: {pet['name']}"
    )
    return jsonify({"success": True})


@app.route("/api/play_with_pet", methods=["POST"])
@requires_unbanned
def play_with_pet_endpoint():
    data = request.get_json()
    pet_id = data.get("pet_id")
    user = Collections["users"].find_one({"username": request.username})
    if not user:
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404
    pet = Collections["pets"].find_one({"id": pet_id})
    if not pet or not pet["alive"]:
        return jsonify({"error": "Pet not found or dead", "code": "pet-not-found"}), 404

    now = time.time()
    last_play_time = pet.get("last_play_time", 0)
    cooldown = 300  # 5 minutes in seconds

    if now - last_play_time < cooldown:
        return (
            jsonify(
                {
                    "error": "Cooldown active",
                    "remaining": cooldown - (now - last_play_time),
                    "code": "cooldown-active",
                }
            ),
            429,
        )

    if pet["personality"] == "Playful":
        pet["happiness"] += 20  # Extra happiness for Playful pets
    elif pet["personality"] == "Adventurous":
        level_up_pet(pet_id, 15)  # Extra experience for Adventurous pets
        pet["happiness"] += 10
    else:
        pet["happiness"] += 10
        
    Collections["pets"].update_one(
        {"id": pet_id},
        {
            "$set": {"last_play_time": now, "happiness": min(100, pet["happiness"])},
        },
    )
    Collections["users"].update_one(
        {"username": request.username}, {"$inc": {"exp": 5}}
    )
    update_account(request.username)
    update_pet(pet_id)
    send_discord_notification(
        "Pet Played With", f"User {request.username} played with pet: {pet['name']}"
    )
    return jsonify({"success": True})


@app.route("/api/mine_tokens", methods=["POST"])
@requires_unbanned
def mine_tokens_endpoint():
    now = time.time()
    user = Collections["users"].find_one({"username": request.username}, {"_id": 0})
    if not user:
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404
    if now - user["last_mine_time"] < TOKEN_MINE_COOLDOWN:
        return (
            jsonify(
                {
                    "error": "Cooldown active",
                    "remaining": TOKEN_MINE_COOLDOWN - (now - user["last_mine_time"]),
                    "code": "cooldown-active",
                }
            ),
            429,
        )

    # Detect token farming patterns
    if user["tokens"] > 10000:  # Arbitrary threshold for excessive tokens
        send_discord_notification(
            "Suspicious Token Mining",
            f"User {request.username} has an unusually high token balance ({user['tokens']}).",
            0xFF0000,
        )

    tokens = random.randint(5, 10)

    extra_tokens = 0
    for pet_id in user.get("pets", []):
        pet = Collections["pets"].find_one({"id": pet_id})
        if pet and pet["alive"]:
            extra_tokens += pet["benefits"].get("token_bonus", 0)
    tokens += extra_tokens

    Collections["users"].update_one(
        {"username": request.username},
        {"$inc": {"tokens": tokens}, "$set": {"last_mine_time": now}},
    )
    Collections["users"].update_one(
        {"username": request.username},
        {"$push": {"history": {"item_id": None, "action": "mine", "timestamp": now}}},
    )
    add_exp(request.username, 5)
    send_discord_notification(
        "Tokens Mined", f"User {request.username} mined {tokens} tokens"
    )
    plugin_manager.trigger_hooks(
        "on_tokens_mined", {"username": request.username, "tokens": tokens}, {"db": db}
    )
    return jsonify({"success": True, "tokens": tokens})


@app.route("/api/market", methods=["GET"])
@requires_unbanned
def market_endpoint():
    items = Collections["items"].find(
        {"for_sale": True, "owner": {"$ne": request.username}},
        {"_id": 0, "item_secret": 0},
    )
    enriched_items = []
    for item in items:
        owner = Collections["users"].find_one({"username": item["owner"]})
        item["ownerPlan"] = get_plan(owner["username"]) if owner else "free"
        enriched_items.append(item)

    # Sort items by plan: Pro+ first, then Pro, then normal users
    enriched_items.sort(
        key=lambda x: ["proplus", "pro", "free"].index(x.get("ownerPlan") or "free")
    )

    return jsonify(enriched_items)


@app.route("/api/sell_item", methods=["POST"])
@requires_unbanned
def sell_item_endpoint():
    data = request.get_json()
    try:
        price = float(data.get("price"))
        if not MIN_ITEM_PRICE <= price <= MAX_ITEM_PRICE:
            raise ValueError
    except ValueError:
        return (
            jsonify({"error": f"Invalid price ({MIN_ITEM_PRICE}-{MAX_ITEM_PRICE})"}),
            400,
        )

    item = Collections["items"].find_one(
        {"id": data.get("item_id"), "owner": request.username}, {"_id": 0}
    )
    if not item:
        return jsonify({"error": "Item not found", "code": "item-not-found"}), 404

    update_data = {
        "for_sale": not item["for_sale"],
        "price": price if not item["for_sale"] else 0,
    }
    Collections["items"].update_one({"id": data.get("item_id")}, {"$set": update_data})
    Collections["users"].update_one(
        {"username": request.username},
        {
            "$push": {
                "history": {
                    "item_id": data.get("item_id"),
                    "action": "sell",
                    "timestamp": time.time(),
                }
            }
        },
    )

    item_name = " ".join(
        [
            item["name"]["adjective"],
            item["name"]["material"],
            item["name"]["noun"],
            item["name"]["suffix"],
            f"#{item['name']['number']}",
        ]
    ).strip()
    send_discord_notification(
        "Item Listed" if update_data["for_sale"] else "Item Unlisted",
        f"User {request.username} {'listed' if update_data["for_sale"] else 'unlisted'} {item_name} {'for ' + str(price) + ' tokens' if update_data['for_sale'] else ''}",
        0xFFFF00,
    )
    return jsonify({"success": True})


@app.route("/api/buy_item", methods=["POST"])
@requires_unbanned
def buy_item_endpoint():
    data = request.get_json()
    item = Collections["items"].find_one(
        {"id": data.get("item_id"), "for_sale": True}, {"_id": 0}
    )
    if not item:
        return jsonify({"error": "Item not available", "code": "item-not-found"}), 404
    if item["owner"] == request.username:
        return (
            jsonify({"error": "Cannot buy own item", "code": "cannot-buy-own-item"}),
            400,
        )

    buyer = Collections["users"].find_one({"username": request.username}, {"_id": 0})
    if buyer["tokens"] < item["price"]:
        return jsonify({"error": "Not enough tokens", "code": "not-enough-tokens"}), 402

    with client.start_session() as session:
        with session.start_transaction():
            Collections["users"].update_one(
                {"username": request.username},
                {"$inc": {"tokens": -item["price"]}},
                session=session,
            )
            Collections["users"].update_one(
                {"username": item["owner"]},
                {"$inc": {"tokens": item["price"]}},
                session=session,
            )
            Collections["users"].update_one(
                {"username": item["owner"]},
                {"$pull": {"items": data.get("item_id")}},
                session=session,
            )
            Collections["users"].update_one(
                {"username": request.username},
                {"$push": {"items": data.get("item_id")}},
                session=session,
            )
            Collections["items"].update_one(
                {"id": data.get("item_id")},
                {"$set": {"owner": request.username, "for_sale": False, "price": 0}},
                session=session,
            )

    Collections["users"].update_one(
        {"username": request.username},
        {
            "$push": {
                "history": {
                    "item_id": data.get("item_id"),
                    "action": "buy",
                    "timestamp": time.time(),
                }
            }
        },
    )
    Collections["users"].update_one(
        {"username": item["owner"]},
        {
            "$push": {
                "history": {
                    "item_id": data.get("item_id"),
                    "action": "sell_complete",
                    "timestamp": time.time(),
                }
            }
        },
    )

    meta = Collections["item_meta"].find_one({"id": item["meta_id"]})
    if meta:
        meta["price_history"].append({"timestamp": time.time(), "price": item["price"]})
        Collections["item_meta"].update_one({"id": item["meta_id"]}, {"$set": meta})

    add_exp(request.username, 5)
    add_exp(item["owner"], 5)

    item_name = " ".join(
        [
            item["name"]["adjective"],
            item["name"]["material"],
            item["name"]["noun"],
            item["name"]["suffix"],
            f"#{item['name']['number']}",
        ]
    ).strip()
    send_discord_notification(
        "Item Purchased",
        f"User {request.username} bought {item_name} from {item['owner']} for {item['price']} tokens",
        0x0000FF,
    )
    plugin_manager.trigger_hooks(
        "on_item_purchase", {"buyer": request.username, "item": item}, {"db": db}
    )
    return jsonify({"success": True})


@app.route("/api/take_item", methods=["POST"])
@requires_unbanned
def take_item_endpoint():
    data = request.get_json()
    item = Collections["items"].find_one({"item_secret": data.get("item_secret")})
    if not item:
        return jsonify({"error": "Invalid secret", "code": "invalid-secret"}), 404

    with client.start_session() as session:
        with session.start_transaction():
            Collections["users"].update_one(
                {"username": item["owner"]},
                {"$pull": {"items": item["id"]}},
                session=session,
            )
            Collections["users"].update_one(
                {"username": request.username},
                {"$push": {"items": item["id"]}},
                session=session,
            )
            Collections["items"].update_one(
                {"item_secret": data.get("item_secret")},
                {"$set": {"owner": request.username, "for_sale": False, "price": 0}},
                session=session,
            )

    Collections["users"].update_one(
        {"username": request.username},
        {
            "$push": {
                "history": {
                    "item_id": item["id"],
                    "action": "take",
                    "timestamp": time.time(),
                }
            }
        },
    )
    Collections["users"].update_one(
        {"username": item["owner"]},
        {
            "$push": {
                "history": {
                    "item_id": item["id"],
                    "action": "taken_from",
                    "timestamp": time.time(),
                }
            }
        },
    )
    return jsonify({"success": True})


@app.route("/api/leaderboard", methods=["GET"])
@requires_unbanned
def leaderboard_endpoint():
    pipeline = [
        {"$match": {"banned": {"$ne": True}}},
        {"$sort": {"tokens": DESCENDING}},
        {"$limit": 10},
        {"$project": {"_id": 0, "username": 1, "tokens": 1}},
    ]
    results = list(Collections["users"].aggregate(pipeline))

    def ordinal(n):
        return "%d%s" % (
            n,
            "tsnrhtdd"[((n // 10 % 10 != 1) * (n % 10 < 4) * n % 10) :: 4],
        )

    for i, item in enumerate(results):
        item["place"] = ordinal(i + 1)
        user = Collections["users"].find_one({"username": item["username"]})
        item["nameplate"] = user.get("equipped_nameplate")
        icon = ""
        if has_proplus(user["username"]):
            icon = "🌟"
        elif has_pro(user["username"]):
            icon = "⭐"
        item["icon"] = icon
    return jsonify({"leaderboard": results})


@app.route("/api/send_message", methods=["POST"])
@requires_unbanned
def send_message_endpoint():
    data = request.get_json()
    plugin_manager.trigger_hooks(
        "on_message_send",
        {"username": request.username, "message": data},
        {"db": db, "chat": {"send_message": send_message}},
    )
    return send_message(
        data.get("room", "global"),
        data.get("message"),
        request.username,
        request.remote_addr,
    )


@app.route("/api/get_messages", methods=["GET"])
@requires_unbanned
def get_messages_endpoint():
    room = request.args.get("room", "global")
    user = Collections["users"].find_one({"username": request.username})

    if not user:
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404

    if not room:
        return (
            jsonify({"error": "Missing room parameter", "code": "missing-parameters"}),
            400,
        )

    if room == "logs" and user["type"] != "admin":
        return jsonify({"error": "You are not an admin", "code": "not-admin"}), 403

    messages = (
        Collections["messages"]
        .find({"room": room}, {"_id": 0})
        .sort("timestamp", ASCENDING)
    )
    visible_messages = []

    for message in messages:
        visibility = message.get("visibility", ["public"])
        if (
            "public" in visibility
            or user["username"] in visibility
            or user["type"] == "admin"
        ):
            visible_messages.append(message)

    return jsonify({"messages": visible_messages})


@app.route("/api/get_banner", methods=["GET"])
@requires_unbanned
def get_banner_endpoint():
    banner = Collections["misc"].find_one({"type": "banner"}, {"_id": 0})
    return jsonify({"banner": banner})


@app.route("/api/stats", methods=["GET"])
def stats_endpoint():
    accounts = list(Collections["users"].find())
    items = list(Collections["items"].find())
    mods = list(Collections["users"].find({"type": "mod"}))
    admins = list(Collections["users"].find({"type": "admin"}))
    users = list(Collections["users"].find({"type": "user"}))
    total_tokens = sum(user["tokens"] for user in accounts)

    return jsonify(
        {
            "stats": {
                "total_tokens": total_tokens,
                "total_accounts": len(accounts),
                "total_items": len(items),
                "total_mods": len(mods),
                "total_admins": len(admins),
                "total_users": len(users),
            }
        }
    )


@app.route("/api/recycle_item", methods=["POST"])
@requires_unbanned
def recycle_endpoint():
    data = request.get_json()
    item_id = data["item_id"]
    item = Collections["items"].find_one({"id": item_id})
    if not item:
        return jsonify({"error": "Item not found"}), 404
    if item["owner"] != request.username:
        return jsonify({"error": "Unauthorized"}), 403

    user = Collections["users"].find_one({"username": request.username})
    if not user:
        return jsonify({"error": "User not found"}), 404

    Collections["users"].update_one(
        {"username": request.username},
        {"$pull": {"items": item_id}, "$inc": {"tokens": 5}},
    )
    Collections["items"].delete_one({"id": item_id})
    return jsonify({"success": True})


@app.route("/api/redeem_creator_code", methods=["POST"])
@requires_unbanned
def redeem_creator_code_endpoint():
    data = request.get_json()
    code = data["code"]

    user = Collections["users"].find_one({"username": request.username})
    if not user:
        return jsonify({"error": "User not found"}), 404

    if user.get("redeemed_creator_code"):
        return jsonify({"error": "Creator code already redeemed"}), 400

    creator_code = Collections["creator_codes"].find_one({"code": code.lower()})
    if not creator_code:
        return jsonify({"error": "Invalid creator code"}), 400

    extra_tokens = creator_code["tokens"]
    extra_pets = creator_code["pets"]

    if extra_pets > 0:
        for _ in range(extra_pets):
            pet = generate_pet(user["username"])
            Collections["pets"].insert_one(pet)
            Collections["users"].update_one(
                {"username": user["username"]}, {"$push": {"pets": pet["id"]}}
            )

    if extra_tokens > 0:
        Collections["users"].update_one(
            {"username": user["username"]}, {"$inc": {"tokens": extra_tokens}}
        )

    Collections["users"].update_one(
        {"username": user["username"]},
        {"$set": {"redeemed_creator_code": True, "creator_code": code}},
    )

    return jsonify({"success": True, "tokens": extra_tokens, "pets": extra_pets})


# Remove all companies and refund owners
def remove_companies():
    companies = Collections["companies"].find()
    for company in companies:
        owner = company["owner"]
        refund_amount = 500 + (company["workers"] * 100)  # Refund base cost + workers
        Collections["users"].update_one(
            {"username": owner}, {"$inc": {"tokens": refund_amount}}
        )
    Collections["companies"].delete_many({})


@app.route("/api/send_tokens", methods=["POST"])
@requires_unbanned
def send_tokens_endpoint():
    data = request.get_json()
    recipient = data.get("recipient")
    amount = data.get("amount")

    if not recipient or not amount:
        return (
            jsonify({"error": "Missing Parameters", "code": "missing-parameters"}),
            400,
        )

    try:
        amount = int(amount)
    except ValueError:
        return jsonify({"error": "Invalid Amount", "code": "invalid-amount"}), 400

    if amount <= 0:
        return jsonify({"error": "Invalid Amount", "code": "invalid-amount"}), 400

    recipient_user = Collections["users"].find_one({"username": recipient})
    if not recipient_user:
        return jsonify({"error": "Recipient not Found", "code": "user-not-found"}), 404

    user = Collections["users"].find_one({"username": request.username})
    if not user:
        return jsonify({"error": "User not Found", "code": "user-not-found"}), 404

    # Check if the sender is a new account
    account_age = time.time() - user["created_at"]
    if (
        account_age < AUTOMOD_CONFIG["NEW_ACCOUNT_AGE_LIMIT"]
        and amount > AUTOMOD_CONFIG["NEW_ACCOUNT_TOKEN_TRANSFER_LIMIT"]
    ):
        return (
            jsonify(
                {
                    "error": "New accounts cannot transfer large amounts of tokens",
                    "code": "new-account-limit",
                }
            ),
            403,
        )

    # Check if the transfer exceeds the global threshold
    if amount > AUTOMOD_CONFIG["TOKEN_TRANSFER_THRESHOLD"]:
        return (
            jsonify(
                {"error": "Transfer exceeds allowed limit", "code": "transfer-limit"}
            ),
            403,
        )

    # Check for suspicious activity
    recent_transfers = Collections["messages"].count_documents(
        {
            "room": "logs",
            "username": request.username,
            "type": "system",
            "timestamp": {
                "$gt": time.time() - AUTOMOD_CONFIG["EXPLOIT_DETECTION_TIME_WINDOW"]
            },
            "message": {"$regex": "sent .* tokens"},
        }
    )
    if recent_transfers >= AUTOMOD_CONFIG["EXPLOIT_DETECTION_THRESHOLD"]:
        return (
            jsonify(
                {"error": "Suspicious activity detected", "code": "suspicious-activity"}
            ),
            403,
        )

    if user["tokens"] < amount:
        return jsonify({"error": "Not enough Tokens", "code": "not-enough-tokens"}), 402

    try:
        Collections["users"].update_one(
            {"username": request.username}, {"$inc": {"tokens": -amount}}
        )
        Collections["users"].update_one(
            {"username": recipient}, {"$inc": {"tokens": amount}}
        )
    except PyMongoError:
        return (
            jsonify({"error": "Internal Server Error", "code": "internal-error"}),
            500,
        )

    send_discord_notification(
        "Tokens Sent", f"{request.username} sent {amount} tokens to {recipient}"
    )

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


@app.route("/api/add_media", methods=["POST"])
@requires_admin
def add_media_endpoint():
    data = request.get_json()
    username = data.get("username")

    if not username:
        return jsonify({"error": "Username not provided"}), 400

    Collections["users"].update_one({"username": username}, {"$set": {"type": "media"}})

    send_discord_notification(
        "Media Added",
        f"Admin {request.username} added {username} as media",
        0x00FF00,
    )

    return jsonify({"success": True})


@app.route("/api/remove_media", methods=["POST"])
@requires_admin
def remove_media_endpoint():
    data = request.get_json()
    username = data.get("username")

    if not username:
        return jsonify({"error": "Username not provided"}), 400

    Collections["users"].update_one({"username": username}, {"$set": {"type": "user"}})

    send_discord_notification(
        "Media Removed",
        f"Admin {request.username} removed {username} as media",
        0x00FF00,
    )

    return jsonify({"success": True})


@app.route("/api/edit_item", methods=["POST"])
@requires_admin
def edit_item_endpoint():
    data = request.get_json()
    return edit_item(
        data.get("item_id"),
        data.get("new_name"),
        data.get("new_icon"),
        data.get("new_rarity"),
    )


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


@app.route("/api/user/<username>", methods=["GET"])
@requires_admin
def user_endpoint(username):
    return get_user(username)


@app.route("/api/delete_message", methods=["POST"])
@requires_mod
def delete_message_endpoint():
    data = request.get_json()
    return delete_message(data.get("message_id"))


@app.route("/api/set_banner", methods=["POST"])
@requires_admin
def set_banner_endpoint():
    data = request.get_json()
    Collections["misc"].delete_many({"type": "banner"})
    Collections["misc"].insert_one({"type": "banner", "value": data.get("banner")})
    send_discord_notification(
        "Banner Updated",
        f"Admin {request.username} set banner to {data.get('banner')}",
        0x00FF00,
    )
    return jsonify({"success": True})


@app.route("/api/get_banned", methods=["GET"])
@requires_admin
def get_banned_endpoint():
    banned = Collections["users"].find({"banned": True}, {"_id": 0})
    return jsonify({"banned_users": [user["username"] for user in banned]})


@app.route("/api/delete_user", methods=["POST"])
@requires_admin
def delete_user_endpoint():
    data = request.get_json()
    username = data.get("username")
    user = Collections["users"].find_one({"username": username})
    if not user:
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404
    Collections["items"].delete_many({"owner": username})
    Collections["users"].delete_one({"username": username})
    Collections["pets"].delete_many({"owner": username})
    Collections["messages"].delete_many({"username": username})

    for sub in user.get("subscriptions", []):
        if sub["status"] == "active":
            try:
                subscription = stripe.Subscription.retrieve(sub["subscription_id"])
                subscription.delete()
            except stripe.error.StripeError as e:
                app.logger.error(f"Error cancelling subscription: {str(e)}")

    send_discord_notification(
        "User deleted", f"Admin {request.username} deleted user {username}", 0xFF0000
    )
    return jsonify({"success": True})


@app.route("/api/give_plan", methods=["POST"])
@requires_admin
def give_plan_endpoint():
    data = request.get_json()
    plan = data.get("plan")
    username = data.get("username")
    length = data.get("length")

    expires = parse_time(length)

    Collections["users"].update_one(
        {"username": username},
        {"$set": {"override_plan": plan, "override_plan_expires": expires}},
    )
    return jsonify({"success": True})


@app.route("/api/remove_plan", methods=["POST"])
@requires_admin
def remove_plan_endpoint():
    data = request.get_json()
    username = data.get("username")
    Collections["users"].update_one(
        {"username": username},
        {"$unset": {"override_plan": None, "override_plan_expires": None}},
    )
    return jsonify({"success": True})


@app.route("/api/create_creator_code", methods=["POST"])
@requires_admin
def create_creator_code_endpoint():
    data = request.get_json()
    code = data.get("code")
    tokens = data.get("tokens")
    pets = data.get("pets")

    Collections["creator_codes"].insert_one(
        {"code": code.lower(), "tokens": tokens, "pets": pets}
    )
    return jsonify({"success": True})


@app.route("/api/delete_creator_code", methods=["POST"])
@requires_admin
def delete_creator_code_endpoint():
    data = request.get_json()
    code = data.get("code")

    Collections["creator_codes"].delete_one({"code": code.lower()})
    return jsonify({"success": True})


@app.route("/api/get_creator_codes", methods=["GET"])
@requires_admin
def get_creator_codes_endpoint():
    codes = Collections["creator_codes"].find({}, {"_id": 0})
    return jsonify({"creator_codes": [code for code in codes]})


@app.route("/api/logs", methods=["GET"])
@requires_admin
def get_logs():
    with open("app.log", "r") as f:
        return f.read()


@app.route("/api/restore_pet", methods=["POST"])
@requires_admin
def restore_pet_endpoint():
    data = request.get_json()
    pet_id = data.get("pet_id")

    pet = Collections["pets"].find_one({"id": pet_id})
    if not pet:
        return jsonify({"error": "Pet not found", "code": "pet-not-found"}), 404

    Collections["pets"].update_one(
        {"id": pet_id}, {"$set": {"alive": True, "happiness": 100, "hunger": 100}}
    )

    return jsonify({"success": True})


@app.route("/api/ping", methods=["GET"])
def ping():
    return "200 OK"


@app.route("/api/auctions", methods=["GET"])
@requires_unbanned
def get_auctions():
    now = time.time()
    expired_auctions = Collections["auctions"].find(
        {"created_at": {"$lte": now - 60 * 60 * 24}}
    )
    for auction in expired_auctions:
        item_id = auction["itemId"]
        owner = auction["owner"]
        highest_bidder = auction.get("currentBidder")
        bid_amount = auction.get("currentBid", 0)

        if highest_bidder:
            # Transfer the item to the highest bidder
            with client.start_session() as session:
                with session.start_transaction():
                    Collections["users"].update_one(
                        {"username": owner},
                        {"$pull": {"items": item_id}},
                        session=session,
                    )
                    Collections["users"].update_one(
                        {"username": highest_bidder},
                        {"$push": {"items": item_id}},
                        session=session,
                    )
                    Collections["items"].update_one(
                        {"id": item_id},
                        {
                            "$set": {
                                "owner": highest_bidder,
                                "for_sale": False,
                                "price": 0,
                            }
                        },
                        session=session,
                    )
                    Collections["users"].update_one(
                        {"username": owner},
                        {"$inc": {"tokens": bid_amount}},
                        session=session,
                    )
        else:
            # No bids, return the item to the owner
            Collections["items"].update_one(
                {"id": item_id}, {"$set": {"for_sale": False, "price": 0}}
            )

        # Remove the auction
        Collections["auctions"].delete_one({"itemId": item_id})

        # Notify via Discord
        send_discord_notification(
            "Auction Ended",
            f"Auction for item {item_id} has ended. {'No bids were placed.' if not highest_bidder else f'{highest_bidder} won the auction with a bid of {bid_amount} tokens.'}",
            0x00FF00,
        )

    auctions = list(Collections["auctions"].find({}, {"_id": 0}))
    return jsonify({"auctions": auctions})


@app.route("/api/create_auction", methods=["POST"])
@requires_unbanned
def create_auction():
    data = request.get_json()
    item_id = data.get("itemId")
    starting_bid = data.get("startingBid")

    starting_bid = float(starting_bid) if starting_bid else 0
    if not item_id or not starting_bid:
        return (
            jsonify(
                {"error": "Missing itemId or startingBid", "code": "missing-parameters"}
            ),
            400,
        )

    # Validate starting bid
    if starting_bid <= 0:
        return jsonify({"error": "Invalid starting bid"}), 400

    if starting_bid > MAX_ITEM_PRICE:
        return jsonify({"error": "Starting bid exceeds maximum price"}), 400

    item = Collections["items"].find_one({"id": item_id, "owner": request.username})
    if not item:
        return jsonify({"error": "Item not found or unauthorized"}), 404

    Collections["auctions"].insert_one(
        {
            "itemId": item_id,
            "itemName": item["name"],
            "itemRarity": {"level": item["level"], "rarity": item["rarity"]},
            "currentBid": starting_bid,
            "owner": request.username,
            "bids": [],
            "created_at": time.time(),
        }
    )
    return jsonify({"success": True})


@app.route("/api/place_bid", methods=["POST"])
@requires_unbanned
def place_bid():
    data = request.get_json()
    item_id = data.get("itemId")
    bid_amount = float(data.get("bidAmount"))

    auction = Collections["auctions"].find_one({"itemId": item_id})
    if not auction:
        return jsonify({"error": "Auction not found"}), 404
    if bid_amount <= auction["currentBid"]:
        return jsonify({"error": "Bid must be higher than the current bid"}), 400

    user = Collections["users"].find_one({"username": request.username})
    if user["tokens"] < bid_amount:
        return jsonify({"error": "Not enough tokens"}), 402

    # Refund previous bidder
    if "currentBidder" in auction:
        Collections["users"].update_one(
            {"username": auction["currentBidder"]},
            {"$inc": {"tokens": auction["currentBid"]}},
        )

    # Deduct tokens from the new bidder
    Collections["users"].update_one(
        {"username": request.username}, {"$inc": {"tokens": -bid_amount}}
    )

    # Update auction
    Collections["auctions"].update_one(
        {"itemId": item_id},
        {"$set": {"currentBid": bid_amount, "currentBidder": request.username}},
    )

    return jsonify({"success": True})


@app.route("/api/stop_auction", methods=["POST"])
@requires_unbanned
def stop_auction():
    data = request.get_json()
    item_id = data.get("itemId")

    auction = Collections["auctions"].find_one({"itemId": item_id})
    user = Collections["users"].find_one({"username": request.username})
    if not auction:
        return jsonify({"error": "Auction not found"}), 404
    if auction["owner"] != request.username and user["type"] != "admin":
        return jsonify({"error": "Unauthorized"}), 403

    # Check if there is a valid bid
    if "currentBidder" not in auction or auction["currentBid"] <= 0:
        # No valid bids, return the item to the owner
        Collections["auctions"].delete_one({"itemId": item_id})
        return jsonify({"success": True, "message": "Auction stopped. No bids placed."})

    # Transfer the item to the highest bidder
    highest_bidder = auction["currentBidder"]
    bid_amount = auction["currentBid"]
    owner = auction["owner"]

    with client.start_session() as session:
        with session.start_transaction():
            # Deduct the item from the owner's inventory
            Collections["users"].update_one(
                {"username": owner}, {"$pull": {"items": item_id}}, session=session
            )

            # Add the item to the highest bidder's inventory
            Collections["users"].update_one(
                {"username": highest_bidder},
                {"$push": {"items": item_id}},
                session=session,
            )

            # Update the item's owner
            Collections["items"].update_one(
                {"id": item_id},
                {"$set": {"owner": highest_bidder, "for_sale": False, "price": 0}},
                session=session,
            )

            # Transfer the bid amount to the auction owner
            Collections["users"].update_one(
                {"username": owner}, {"$inc": {"tokens": bid_amount}}, session=session
            )

    # Remove the auction
    Collections["auctions"].delete_one({"itemId": item_id})

    # Notify via Discord
    send_discord_notification(
        "Auction Stopped",
        f"Auction for item {item_id} has ended. {highest_bidder} won the auction with a bid of {bid_amount} tokens. {owner} received the tokens.",
        0x00FF00,
    )

    return jsonify(
        {"success": True, "message": f"Auction stopped. {highest_bidder} won the item."}
    )


@app.route("/api/delete_auction", methods=["POST"])
@requires_admin
def delete_auction():
    data = request.get_json()
    item_id = data.get("itemId")

    auction = Collections["auctions"].find_one({"itemId": item_id})
    if not auction:
        return jsonify({"error": "Auction not found"}), 404

    Collections["auctions"].delete_one({"itemId": item_id})

    return jsonify({"success": True, "message": "Auction deleted."})


# Submit a User Report
@app.route("/api/report_user", methods=["POST"])
@requires_unbanned
def report_user_endpoint():
    data = request.get_json()
    username = data.get("username")
    comment = data.get("comment")

    if not username or not comment:
        return jsonify({"error": "Missing username or comment"}), 400

    report = {
        "id": str(uuid4()),
        "username": username,
        "comment": comment,
        "reportedBy": request.username,
        "timestamp": int(time.time()),
        "status": "pending",
    }
    Collections["reports"].insert_one(report)
    return jsonify({"success": True})


# Fetch All Reports (Admin Only)
@app.route("/api/reports", methods=["GET"])
@requires_admin
def get_reports_endpoint():
    reports = list(Collections["reports"].find({"status": "pending"}, {"_id": 0}))
    return jsonify({"reports": reports})


# Handle Report Actions
@app.route("/api/handle_report", methods=["POST"])
@requires_admin
def handle_report_endpoint():
    data = request.get_json()
    report_id = data.get("reportId")
    action = data.get("action")
    duration = data.get("duration")
    reason = data.get("reason")

    report = Collections["reports"].find_one({"id": report_id})
    if not report:
        return jsonify({"error": "Report not found"}), 404

    if action == "ban":
        if not duration or not reason:
            return jsonify({"error": "Missing duration or reason for ban"}), 400
        ban_user(report["username"], duration, reason)
    elif action == "mute":
        if not duration:
            return jsonify({"error": "Missing duration for mute"}), 400
        mute_user(report["username"], duration)
    elif action == "cancel":
        pass  # No additional action needed for cancel
    else:
        return jsonify({"error": "Invalid action"}), 400

    Collections["reports"].update_one({"id": report_id}, {"$set": {"status": action}})
    return jsonify({"success": True})


@app.route("/api/get_user_data", methods=["POST"])
@requires_admin
def get_user_data_endpoint():
    data = request.get_json()
    username = data.get("username")

    if not username:
        return jsonify({"error": "Username not provided"}), 400

    user = Collections["users"].find_one({"username": username})
    if not user:
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404

    # Find other users created with the same IP
    same_ip_users = list(
        Collections["users"].find(
            {"creation_ip": user.get("creation_ip"), "username": {"$ne": username}},
            {"_id": 0, "username": 1},
        )
    )

    # Prepare response
    user_data = {
        "username": user["username"],
        "ip": user.get("creation_ip"),
        "creation_time": user["created_at"],
        "punishment_history": {
            "banned": user.get("banned"),
            "banned_until": user.get("banned_until"),
            "banned_reason": user.get("banned_reason"),
            "muted": user.get("muted"),
            "muted_until": user.get("muted_until"),
        },
        "history": user.get("history", []),
        "tokens": user["tokens"],
        "level": user.get("level"),
        "exp": user.get("exp"),
        "plan_status": get_plan(username),
        "items_count": len(user.get("items", [])),
        "pets_count": len(user.get("pets", [])),
        "other_users_same_ip": [u["username"] for u in same_ip_users],
    }

    return jsonify({"success": True, "user_data": user_data})


@app.route("/api/add_gems", methods=["POST"])
@requires_admin
def add_gems_endpoint():
    data = request.get_json()
    username = data.get("username")
    gems = data.get("gems")

    if not username or gems is None:
        return jsonify({"error": "Missing username or gems"}), 400

    try:
        gems = int(gems)
        if gems <= 0:
            raise ValueError
    except ValueError:
        return jsonify({"error": "Invalid gems value"}), 400

    user = Collections["users"].find_one({"username": username})
    if not user:
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404

    Collections["users"].update_one({"username": username}, {"$inc": {"gems": gems}})
    send_discord_notification(
        "Gems Added",
        f"Admin {request.username} added {gems} gems to {username}",
        0x00FF00,
    )
    return jsonify({"success": True})


@app.route("/api/remove_gems", methods=["POST"])
@requires_admin
def remove_gems_endpoint():
    data = request.get_json()
    username = data.get("username")
    gems = data.get("gems")

    if not username or gems is None:
        return jsonify({"error": "Missing username or gems"}), 400

    try:
        gems = int(gems)
        if gems <= 0:
            raise ValueError
    except ValueError:
        return jsonify({"error": "Invalid gems value"}), 400

    user = Collections["users"].find_one({"username": username})
    if not user:
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404

    if user.get("gems", 0) < gems:
        return jsonify({"error": "Not enough gems to remove"}), 400

    Collections["users"].update_one({"username": username}, {"$inc": {"gems": -gems}})
    send_discord_notification(
        "Gems Removed",
        f"Admin {request.username} removed {gems} gems from {username}",
        0xFF0000,
    )
    return jsonify({"success": True})


@app.route("/api/set_gems", methods=["POST"])
@requires_admin
def set_gems_endpoint():
    data = request.get_json()
    username = data.get("username")
    gems = data.get("gems")

    if not username or gems is None:
        return jsonify({"error": "Missing username or gems"}), 400

    if gems != "$INFINITY":
        try:
            gems = int(gems)
            if gems < 0:
                raise ValueError
        except ValueError:
            return jsonify({"error": "Invalid gems value"}), 400

    user = Collections["users"].find_one({"username": username})
    if not user:
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404

    Collections["users"].update_one({"username": username}, {"$set": {"gems": gems}})
    send_discord_notification(
        "Gems Set",
        f"Admin {request.username} set {username}'s gems to {gems}",
        0x00FF00,
    )
    return jsonify({"success": True})


@app.route("/api/buy_cosmetic", methods=["POST"])
@requires_unbanned
def buy_cosmetic_endpoint():
    data = request.get_json()
    cosmetic_id = data.get("cosmetic_id")

    if not cosmetic_id or cosmetic_id not in COSMETICS:
        return (
            jsonify({"error": "Invalid cosmetic ID", "code": "invalid-cosmetic-id"}),
            400,
        )

    cosmetic = COSMETICS[cosmetic_id]
    user = Collections["users"].find_one({"username": request.username})

    if not cosmetic["price"]:
        return jsonify({"error": "Cosmetic not for sale", "code": "not-for-sale"}), 400

    if not user:
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404

    if user["gems"] != "$INFINITY":
        if user["gems"] < cosmetic["price"]:
            return jsonify({"error": "Not enough gems", "code": "not-enough-gems"}), 402

    if "cosmetics" in user and cosmetic_id in user["cosmetics"]:
        return (
            jsonify({"error": "Cosmetic already owned", "code": "already-owned"}),
            400,
        )

    Collections["users"].update_one(
        {"username": request.username},
        {
            "$addToSet": {"cosmetics": cosmetic_id},
        },
    )

    if user["gems"] != "$INFINITY":
        Collections["users"].update_one(
            {"username": request.username},
            {"$inc": {"gems": -cosmetic["price"]}},
        )

    send_discord_notification(
        "Cosmetic Purchased",
        f"User {request.username} purchased cosmetic: {cosmetic['name']} for {cosmetic['price']} gems",
        0x00FF00,
    )

    return jsonify({"success": True, "cosmetic": cosmetic})


@app.route("/api/get_cosmetics", methods=["GET"])
@requires_unbanned
def get_cosmetics_endpoint():
    user = Collections["users"].find_one({"username": request.username})
    if not user:
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404

    owned_cosmetic_ids = user.get("cosmetics", [])
    owned_cosmetics = [
        {"id": cid, **COSMETICS[cid]} for cid in owned_cosmetic_ids if cid in COSMETICS
    ]
    available_cosmetics = [
        {"id": cid, **COSMETICS[cid]}
        for cid in COSMETICS
        if (cid not in owned_cosmetics and COSMETICS[cid]["price"])
    ]

    equipped_cosmetics = {
        "messageplate": COSMETICS.get(user.get("equipped_messageplate")),
        "nameplate": COSMETICS.get(user.get("equipped_nameplate")),
    }

    return jsonify(
        {
            "owned": owned_cosmetics,
            "cosmetics": available_cosmetics,
            "equipped": equipped_cosmetics,
        }
    )


@app.route("/api/equip_cosmetic", methods=["POST"])
@requires_unbanned
def equip_cosmetic_endpoint():
    data = request.get_json()
    cosmetic_id = data.get("cosmetic_id")

    if not cosmetic_id:
        return (
            jsonify({"error": "Missing cosmetic ID", "code": "missing-parameters"}),
            400,
        )

    user = Collections["users"].find_one({"username": request.username})
    if not user:
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404

    if cosmetic_id not in COSMETICS:
        return (
            jsonify({"error": "Invalid cosmetic ID", "code": "invalid-cosmetic-id"}),
            400,
        )

    if cosmetic_id not in user.get("cosmetics", []):
        return jsonify({"error": "Cosmetic not owned", "code": "not-owned"}), 400

    cosmetic = COSMETICS[cosmetic_id]
    if cosmetic["type"] == "messageplate":
        Collections["users"].update_one(
            {"username": request.username},
            {"$set": {"equipped_messageplate": cosmetic_id}},
        )
    elif cosmetic["type"] == "nameplate":
        Collections["users"].update_one(
            {"username": request.username},
            {"$set": {"equipped_nameplate": cosmetic_id}},
        )

    send_discord_notification(
        "Cosmetic Equipped",
        f"User {request.username} equipped cosmetic: {cosmetic['name']} ({cosmetic['type']})",
        0x00FF00,
    )

    return jsonify({"success": True, "equipped_cosmetic": cosmetic_id})
  
@app.route("/api/revive_pet", methods=["POST"])
@requires_unbanned
def revive_pet_endpoint():
    data = request.get_json()
    pet_id = data.get("pet_id")

    if not pet_id:
        return jsonify({"error": "Pet ID not provided"}), 400

    pet = Collections["pets"].find_one({"id": pet_id})
    if not pet:
        return jsonify({"error": "Pet not found", "code": "pet-not-found"}), 404
      
    user = Collections["users"].find_one({"username": request.username})
    if not user:
        return jsonify({"error": "User not found", "code": "user-not-found"}), 404
      
    if user["gems"] != "$INFINITY":
        if user["gems"] < 50:
            return jsonify({"error": "Not enough gems", "code": "not-enough-gems"}), 402
          
    if pet["alive"]:
        return jsonify({"error": "Pet is already alive", "code": "pet-alive"}), 400
      
    if pet["owner"] != request.username:
        return jsonify({"error": "Unauthorized", "code": "unauthorized"}), 403
      
    if user["gems"] != "$INFINITY":
        Collections["users"].update_one(
            {"username": request.username}, {"$inc": {"gems": -50}}
        )

    Collections["pets"].update_one(
        {"id": pet_id}, {"$set": {"alive": True, "happiness": 100, "hunger": 100, "last_fed": time.time(), "last_play_time": time.time()}}
    )

    send_discord_notification(
        "Pet Revived",
        f"{request.username} revived pet {pet['name']} ({pet['id']})",
        0x00FF00,
    )

    return jsonify({"success": True})


@app.route("/stripe_webhook", methods=["POST"])
def stripe_webhook():
    payload = request.data
    sig_header = request.headers.get("Stripe-Signature")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        return jsonify({"error": "Invalid payload"}), 400
    except stripe.error.SignatureVerificationError:
        return jsonify({"error": "Invalid signature"}), 400

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        username = session["metadata"].get("username")
        item_key = session["metadata"].get("item")
        subscription_id = session.get("subscription")

        if not username or not item_key:
            return jsonify({"error": "Missing metadata"}), 400

        if subscription_id:
            pending_subscription = Collections["pending_subscriptions"].find_one(
                {"subscription_id": subscription_id}
            )
            if not pending_subscription:
                return jsonify({"error": "Pending subscription not found"}), 404
            Collections["pending_subscriptions"].delete_one(
                {"subscription_id": subscription_id}
            )

            Collections["users"].update_one(
                {"username": username},
                {"$addToSet": {"subscriptions": pending_subscription}},
            )
        else:
            # One-time gem purchase
            gems_lookup = {
                "gems_500": 500,
                "gems_1000": 1000,
                "gems_2500": 2500,
                "gems_5000": 5000,
            }
            gems = gems_lookup.get(item_key)

            if not gems:
                return jsonify({"error": "Unknown gem pack"}), 400

            Collections["users"].update_one(
                {"username": username},
                {"$inc": {"gems": gems}},
            )

            send_discord_notification(
                "Gem Purchase", f"{username} purchased {gems} gems."
            )

    elif event["type"] == "customer.subscription.created":
        subscription = event["data"]["object"]

        price_id = subscription["items"]["data"][0]["price"]["id"]
        price = stripe.Price.retrieve(price_id)
        product = stripe.Product.retrieve(price["product"])
        product_name = product.name.lower()

        if "pro_plus" in product_name or "proplus" in product_name:
            plan = "proplus"
        elif "pro" in product_name:
            plan = "pro"
        else:
            plan = "unknown"

        Collections["pending_subscriptions"].insert_one(
            {
                "subscription_id": subscription["id"],
                "price_id": price.id,
                "product": product.name,
                "interval": price.recurring.interval,
                "status": subscription.status,
                "current_period_end": subscription.get("current_period_end"),
                "plan": plan,
            },
        )

    elif event["type"] in [
        "customer.subscription.updated",
        "customer.subscription.deleted",
    ]:
        subscription = event["data"]["object"]
        Collections["users"].update_one(
            {"subscriptions.subscription_id": subscription["id"]},
            {
                "$set": {
                    "subscriptions.$.status": subscription["status"],
                    "subscriptions.$.current_period_end": subscription.get(
                        "current_period_end"
                    ),
                }
            },
        )

    return jsonify({"success": True}), 200


@app.route("/create_checkout_session", methods=["POST"])
def create_checkout_session():
    data = request.get_json()

    item_key = data.get("item")
    username = data.get("username")

    if item_key not in PRICE_IDS:
        return jsonify({"error": "Invalid item type"}), 400
    if not username:
        return jsonify({"error": "Missing username"}), 400

    price_id = PRICE_IDS[item_key]
    is_subscription = item_key.startswith("pro")

    try:
        session_params = {
            "mode": "subscription" if is_subscription else "payment",
            "line_items": [{"price": price_id, "quantity": 1}],
            "success_url": "https://economix.lol",
            "cancel_url": "https://economix.lol",
            "metadata": {"username": username, "item": item_key},
        }
        if not is_subscription:
            session_params["customer_creation"] = "always"

        session = stripe.checkout.Session.create(**session_params)
        return jsonify({"url": session.url})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


class Event:
    def __init__(self, name: str, payload: dict, context: dict):
        self.name = name
        self.payload = payload
        self.context = context


class PluginManager:
    def __init__(self):
        self.hooks: Dict[str, List[Callable]] = {}

    def register_hook(self, event_name: str, callback: Callable):
        if event_name not in self.hooks:
            self.hooks[event_name] = []
        self.hooks[event_name].append(callback)

    def trigger_hooks(self, event_name: str, payload: dict, context: dict):
        event = Event(name=event_name, payload=payload, context=context)
        for callback in self.hooks.get(event_name, []):
            callback(event)

    def load_plugins(self, plugins_dir: str):
        try:
            for filename in os.listdir(plugins_dir):
                if filename.endswith(".py"):
                    plugin_path = os.path.join(plugins_dir, filename)
                    spec = importlib.util.spec_from_file_location(
                        filename[:-3], plugin_path
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if hasattr(module, "register"):
                        module.register(self)
        except Exception as e:
            app.logger.error(f"Error loading plugins: {e}")


# Initialize PluginManager and load plugins
plugin_manager = PluginManager()
plugin_manager.load_plugins(os.path.join(os.path.dirname(__file__), "plugins"))
