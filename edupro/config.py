"""
config.py
=========
Central configuration for the EduPro course-demand and revenue-forecasting project.

Everything that a reviewer might reasonably want to change (file paths, the length of
the forecast window, the model hyper-parameters) lives here, so the analysis code
below never hard-codes a magic number.
"""

from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
TABLE_DIR = OUTPUT_DIR / "tables"

# The workbook downloaded from the project brief. Put it in ./data/
WORKBOOK_NAME = "EduPro Online Platform.xlsx"
WORKBOOK_PATH = DATA_DIR / WORKBOOK_NAME

for _d in (DATA_DIR, OUTPUT_DIR, FIGURE_DIR, MODEL_DIR, TABLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------
# Sheet discovery
# --------------------------------------------------------------------------------------
# The workbook has four tabs. We match them case-insensitively and tolerate small
# naming differences ("Course" vs "Courses", "Txns" vs "Transactions").
SHEET_ALIASES = {
    "users": ["users", "user", "learners", "students"],
    "teachers": ["teachers", "teacher", "instructors", "instructor"],
    "courses": ["courses", "course", "catalog", "catalogue"],
    "transactions": ["transactions", "transaction", "txn", "txns", "sales", "payments"],
}

# --------------------------------------------------------------------------------------
# Column aliases
# --------------------------------------------------------------------------------------
# Keys are the canonical names used throughout the codebase. Values are the raw
# spellings we are willing to accept. Matching ignores case, spaces and underscores,
# so "course_id", "Course ID" and "CourseID" all resolve to "CourseID".
COLUMN_ALIASES = {
    "courses": {
        "CourseID": ["courseid", "course", "cid"],
        "CourseName": ["coursename", "coursetitle", "title", "name"],
        "CourseCategory": ["coursecategory", "category", "subject", "domain"],
        "CourseType": ["coursetype", "type", "format", "deliverymode"],
        "CourseLevel": ["courselevel", "level", "difficulty"],
        "CoursePrice": ["courseprice", "price", "fee", "cost"],
        "CourseDuration": ["courseduration", "duration", "hours", "lengthhours"],
        "CourseRating": ["courserating", "rating", "avgrating", "averagerating"],
        "TeacherID": ["teacherid", "instructorid", "tid"],
    },
    "teachers": {
        "TeacherID": ["teacherid", "instructorid", "tid", "teacher"],
        "TeacherName": ["teachername", "instructorname", "name"],
        "Expertise": ["expertise", "specialisation", "specialization", "subject", "domain"],
        "YearsOfExperience": [
            "yearsofexperience", "experience", "experienceyears", "yearsexperience", "yoe",
        ],
        "TeacherRating": ["teacherrating", "rating", "instructorrating"],
        "CourseID": ["courseid"],
    },
    "transactions": {
        "TransactionID": ["transactionid", "txnid", "id", "orderid"],
        "CourseID": ["courseid", "course", "cid"],
        "UserID": ["userid", "learnerid", "studentid"],
        "TransactionDate": ["transactiondate", "date", "purchasedate", "enrollmentdate", "timestamp"],
        "Amount": ["amount", "revenue", "value", "paid", "price", "amountpaid"],
    },
    "users": {
        "UserID": ["userid", "learnerid", "studentid"],
        "UserName": ["username", "name"],
        "Age": ["age"],
        "Gender": ["gender", "sex"],
        "Email": ["email", "emailaddress"],
    },
}

# --------------------------------------------------------------------------------------
# Temporal design
# --------------------------------------------------------------------------------------
# The forecasting model is trained on a HISTORY window and scored on a FORECAST window.
# Features never see a transaction dated on or after the cutoff, which is what stops
# "past enrolment count" from leaking the answer into "future enrolment count".
FORECAST_WINDOW_MONTHS = 3          # length of the held-out future period
MIN_HISTORY_MONTHS = 6              # refuse to build the temporal model below this

# --------------------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------------------
PRICE_BANDS = ["Low", "Medium", "High"]                 # tertiles of CoursePrice
DURATION_BUCKETS = ["Short", "Medium", "Long", "Extended"]   # quartiles of CourseDuration
RATING_TIERS = ["Poor", "Average", "Good", "Excellent"]      # fixed cuts on CourseRating
RATING_TIER_EDGES = [0.0, 3.0, 3.75, 4.5, 5.01]
EXPERIENCE_BUCKETS = ["Junior", "Mid", "Senior", "Veteran"]
EXPERIENCE_EDGES = [-0.01, 3, 7, 12, 1000]

# Ordinal encoding for course level (unrecognised labels fall back to the median).
LEVEL_ORDER = {"beginner": 1, "basic": 1, "foundation": 1,
               "intermediate": 2, "medium": 2,
               "advanced": 3, "expert": 3, "professional": 3}

# Drop one of any pair of numeric features correlated above this threshold.
CORRELATION_THRESHOLD = 0.95

# --------------------------------------------------------------------------------------
# Modelling
# --------------------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

MODEL_GRID = {
    "Ridge": {"alpha": [0.1, 1.0, 10.0, 100.0]},
    "Lasso": {"alpha": [0.001, 0.01, 0.1, 1.0]},
    "RandomForest": {
        "n_estimators": [300],
        "max_depth": [None, 8, 14],
        "min_samples_leaf": [1, 2, 4],
    },
    "GradientBoosting": {
        "n_estimators": [200, 400],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3],
    },
}

# Targets modelled at course level.
TARGETS = ["EnrollmentCount", "CourseRevenue"]
