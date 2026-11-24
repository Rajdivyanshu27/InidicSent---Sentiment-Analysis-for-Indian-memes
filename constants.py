import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(APP_DIR)
ARTIFACTS_DIR = os.path.join(APP_DIR, "artifacts")
STATIC_DIR = os.path.join(APP_DIR, "static")

TRAIN_CSV = os.path.join(DATA_DIR, "memotion3", "train.csv")
VAL_CSV = os.path.join(DATA_DIR, "val.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

IMAGE_DIRS = {
    "train": os.path.join(DATA_DIR, "trainImages"),
    "val": os.path.join(DATA_DIR, "valImages"),
    "test": os.path.join(DATA_DIR, "testImages"),
}

SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]
SENTIMENT_RAW_TO_ID = {
    "very_negative": 0,
    "negative": 0,
    "neutral": 1,
    "positive": 2,
    "very_positive": 2,
}

EMOTION_KEYS = ["humour", "sarcastic", "offensive", "motivational"]
EMOTION_DISPLAY = {
    "humour": "Humorous",
    "sarcastic": "Sarcastic",
    "offensive": "Offensive",
    "motivational": "Motivational",
}

TASK_B_NEGATIVE_RAW = {
    "humour": "not_funny",
    "sarcastic": "not_sarcastic",
    "offensive": "not_offensive",
    "motivational": "not_motivational",
}

TASK_C_SCALES = {
    "humour": {
        "raw_to_id": {"not_funny": 0, "funny": 1, "very_funny": 2, "hilarious": 3},
        "labels": ["Not Funny", "Funny", "Very Funny", "Hilarious"],
    },
    "sarcastic": {
        "raw_to_id": {
            "not_sarcastic": 0,
            "general": 1,
            "twisted_meaning": 2,
            "very_twisted": 3,
        },
        "labels": ["Not Sarcastic", "General", "Twisted Meaning", "Very Twisted"],
    },
    "offensive": {
        "raw_to_id": {
            "not_offensive": 0,
            "slight": 1,
            "very_offensive": 2,
            "hateful_offensive": 3,
        },
        "labels": ["Not Offensive", "Slight", "Very Offensive", "Hateful Offensive"],
    },
    "motivational": {
        "raw_to_id": {"not_motivational": 0, "motivational": 1},
        "labels": ["Not Motivational", "Motivational"],
    },
}
