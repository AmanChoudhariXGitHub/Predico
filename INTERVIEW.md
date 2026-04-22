# Predico — Complete Interview Preparation Guide

> **Project:** Predico — General Disease Prediction Web Application
> **Stack:** Django · PostgreSQL · Machine Learning (Random Forest) · Python · HTML/CSS/JS
> **Deployment:** Render / Vercel / Neon DB
> **Dataset:** Kaggle — Disease Prediction Using Machine Learning

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Database Design & Models](#3-database-design--models)
4. [Machine Learning Pipeline](#4-machine-learning-pipeline)
5. [Backend & API Design](#5-backend--api-design)
6. [Authentication & Authorization](#6-authentication--authorization)
7. [Deployment & DevOps](#7-deployment--devops)
8. [Code Deep-Dives & Gotchas](#8-code-deep-dives--gotchas)
9. [Interview Q&A — Software Developer / Backend Developer](#9-interview-qa--software-developer--backend-developer)
10. [Interview Q&A — Data Scientist / ML Engineer](#10-interview-qa--data-scientist--ml-engineer)
11. [Interview Q&A — Full Stack Developer](#11-interview-qa--full-stack-developer)
12. [Interview Q&A — System Design](#12-interview-qa--system-design)
13. [Known Issues & How You'd Fix Them](#13-known-issues--how-youd-fix-them)
14. [Improvements & Extensions to Mention](#14-improvements--extensions-to-mention)

---

## 1. Project Overview

**What is Predico?**
Predico is a full-stack web application that allows patients to input their symptoms and receive a predicted disease diagnosis powered by a pre-trained machine learning model (Random Forest Classifier). The app also connects patients to appropriate medical specialists and enables doctor-patient consultations via a built-in chat system.

**Core User Roles:**
- **Admin** — Django superuser; views all feedback, manages the platform
- **Patient** — Signs up, selects symptoms, gets disease predictions, consults doctors, rates doctors
- **Doctor** — Signs up with medical credentials, receives consultation requests, chats with patients, closes consultations

**Core Features:**
- Symptom-based disease prediction (41 diseases, 132 symptoms)
- ML confidence scoring
- Doctor recommendation by specialization
- Consultation booking and real-time chat
- Doctor rating & review system
- Patient and doctor profile management
- Admin feedback dashboard

---

## 2. System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT (Browser)                         │
│              HTML Templates + JavaScript (AJAX/Fetch)           │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP Requests
┌───────────────────────────▼─────────────────────────────────────┐
│                    Django Web Server (Gunicorn)                  │
│                                                                  │
│   ┌─────────────────┐  ┌─────────────────┐  ┌───────────────┐  │
│   │   main_app      │  │    accounts     │  │    chats      │  │
│   │  (core logic)   │  │  (auth/users)   │  │ (chat/feedback│  │
│   └────────┬────────┘  └────────┬────────┘  └───────┬───────┘  │
│            │                    │                    │          │
│   ┌────────▼────────────────────▼────────────────────▼───────┐  │
│   │                    Django ORM                             │  │
│   └────────────────────────┬──────────────────────────────────┘  │
│                            │                                  │
│   ┌────────────────────────▼──────────────────────────────────┐  │
│   │               PostgreSQL Database                         │  │
│   └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│   ┌───────────────────────────────────────────────────────────┐  │
│   │        ML Layer — Joblib-loaded Random Forest Model       │  │
│   │               (trained_model file on disk)                │  │
│   └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Django App Structure

```
disease_prediction/          ← Django project root (settings, urls, wsgi)
├── main_app/                ← Core app: disease prediction, consultations, ratings
├── accounts/                ← Authentication: patient/doctor signup, login, profile edit
├── chats/                   ← Chat system and feedback
├── Accuracy/                ← Standalone ML training scripts (not part of Django)
├── templates/               ← HTML templates (served by WhiteNoise in production)
├── trained_model            ← Serialized Random Forest model (joblib)
└── manage.py
```

### Request-Response Workflow (Disease Prediction)

```
Patient selects symptoms (frontend checkboxes)
        │
        ▼
POST /checkdisease  (AJAX with symptoms[] array)
        │
        ▼
Django view: checkdisease()
        │
        ├── Builds 132-element binary feature vector
        │
        ├── model.predict(inputtest)  → predicted disease
        │
        ├── model.predict_proba()     → confidence score
        │
        ├── Maps disease → specialist type
        │
        ├── Saves diseaseinfo record to DB
        │
        └── Returns JsonResponse {predicteddisease, confidencescore, consultdoctor}
                │
                ▼
        Frontend displays result → Patient navigates to doctor list
```

---

## 3. Database Design & Models

### Entity-Relationship Overview

```
User (Django built-in)
 ├── OneToOne ──► patient
 │                  │
 │                  ├── ForeignKey ──► diseaseinfo (many per patient)
 │                  │                      │
 │                  │                      └── OneToOne ──► consultation
 │                  │                                            │
 │                  ├── ForeignKey ──► consultation              │
 │                  │                                            │
 │                  └── ForeignKey ──► rating_review             │
 │                                                               │
 └── OneToOne ──► doctor ◄── ForeignKey ──────────────consultation
                    │
                    └── ForeignKey ──► rating_review
```

### Model Details

**patient**
| Field | Type | Notes |
|---|---|---|
| user | OneToOneField(User) | Primary key — links to Django auth |
| is_patient | BooleanField | Default True, used for role-check |
| is_doctor | BooleanField | Default False |
| name, dob, address, mobile_no, gender | CharField/DateField | Profile fields |
| age (property) | Computed | Calculates age from dob dynamically |

**doctor**
| Field | Type | Notes |
|---|---|---|
| user | OneToOneField(User) | Primary key |
| registration_no, year_of_registration | CharField/DateField | Medical credentials |
| qualification, State_Medical_Council, specialization | CharField | Professional info |
| rating | IntegerField | Aggregated from rating_review |

**diseaseinfo**
| Field | Type | Notes |
|---|---|---|
| patient | ForeignKey(patient) | SET_NULL on delete |
| diseasename | CharField | ML prediction output |
| no_of_symp | IntegerField | Count of selected symptoms |
| symptomsname | ArrayField(CharField) | PostgreSQL-specific array field |
| confidence | DecimalField | Prediction confidence % |
| consultdoctor | CharField | Mapped specialist type |

**consultation**
| Field | Type | Notes |
|---|---|---|
| patient | ForeignKey(patient) | SET_NULL on delete |
| doctor | ForeignKey(doctor) | SET_NULL on delete |
| diseaseinfo | OneToOneField(diseaseinfo) | Linked prediction record |
| consultation_date | DateField | Auto-set to today |
| status | CharField | "active" or "closed" |

**Chat**
| Field | Type | Notes |
|---|---|---|
| consultation_id | ForeignKey(consultation) | Which consultation this message belongs to |
| sender | ForeignKey(User) | Could be patient or doctor |
| message | TextField | Chat message content |
| created | DateTimeField | Auto-set on creation |

**rating_review**
| Field | Type | Notes |
|---|---|---|
| patient | ForeignKey(patient) | Who submitted the review |
| doctor | ForeignKey(doctor) | Who was reviewed |
| rating | IntegerField | Numeric score |
| review | TextField | Optional text |
| rating_is (property) | Computed | Calculates average doctor rating |

### Key Design Decisions to Discuss

- `ArrayField` from `django.contrib.postgres.fields` — requires PostgreSQL, not SQLite-compatible
- `OneToOneField` with `primary_key=True` on patient/doctor — the User IS the patient/doctor, no separate ID
- `SET_NULL` on ForeignKeys in diseaseinfo/consultation — patient/doctor deletion doesn't cascade-destroy medical records
- The `rating` field on `doctor` is denormalized — it's updated via `doctor.objects.filter().update()` whenever a review is submitted, avoiding recalculation on every read

---

## 4. Machine Learning Pipeline

### Dataset

- **Source:** Kaggle — "Disease Prediction Using Machine Learning" by Neelima
- **Structure:** CSV with 132 binary symptom columns + 1 target column `prognosis`
- **Target classes:** 41 diseases
- **Features:** Binary (0/1) — symptom present or not

### Training Pipeline (Accuracy/model.py)

```python
# 1. Load data
data = pd.read_csv("Testing.csv")
X = data.drop(columns=['prognosis'])
y = data['prognosis']

# 2. Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 3. Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2,
    random_state=42, stratify=y_resampled
)

# 4. Train Random Forest
model = RandomForestClassifier(random_state=42)

# 5. Hyperparameter tuning with GridSearchCV + StratifiedKFold
param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20, 30]}
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=cv_strategy, scoring="accuracy", n_jobs=-1)
```

### Inference in Production (main_app/views.py)

```python
# On server startup — model loaded ONCE (module-level, not per-request)
import joblib as jb
model = jb.load('trained_model')

# Per prediction request:
testingsymptoms = [0] * 132           # 132 binary features
for k, symptom in enumerate(symptomslist):
    if symptom in psymptoms:
        testingsymptoms[k] = 1

predicted = model.predict([testingsymptoms])
confidence = model.predict_proba([testingsymptoms]).max() * 100
```

### Why Random Forest?

- Handles binary feature spaces well
- Naturally provides `predict_proba()` for confidence scoring
- Resistant to overfitting compared to single decision trees
- No feature scaling required (tree-based)
- Interpretable feature importances

### SMOTE — Synthetic Minority Oversampling Technique

- Creates synthetic samples for minority classes by interpolating between existing minority examples
- Used here to ensure the model doesn't become biased toward more common diseases
- Applied BEFORE the train/test split to avoid data leakage in the training set

### Disease-to-Specialist Mapping Logic

The `checkdisease` view contains an if/elif chain mapping predicted diseases to doctor specializations:

```python
Rheumatologist = ['Osteoarthristis', 'Arthritis']
Cardiologist = ['Heart attack', 'Bronchial Asthma', 'Hypertension ']
Neurologist = ['Varicose veins', 'Paralysis (brain hemorrhage)', 'Migraine', 'Cervical spondylosis']
# ... etc.
```

**Bug to note in interviews:** The first condition for `Rheumatologist` uses `if` instead of `elif`, so if a disease matches `Rheumatologist`, it then CONTINUES checking all `elif` branches. This means a Rheumatology disease could get overwritten by a later match. This is a logical bug worth flagging.

---

## 5. Backend & API Design

### URL Structure

```
/                          → home (main_app)
/admin/                    → Django admin panel
/admin_ui                  → Custom admin dashboard
/patient_ui                → Patient home (profile)
/doctor_ui                 → Doctor home (profile)
/checkdisease              → GET: symptom form | POST: AJAX prediction endpoint
/consult_a_doctor          → List of doctors
/make_consultation/<str>   → POST: Book consultation
/consultationview/<int>    → View consultation + chat
/close_consultation/<int>  → POST: Doctor closes consultation
/rate_review/<int>         → POST: Patient rates doctor
/pconsultation_history     → Patient's past consultations
/dconsultation_history     → Doctor's past consultations
/pviewprofile/<str>        → Patient profile view
/dviewprofile/<str>        → Doctor profile view
/savepdata/<str>           → POST: Save patient profile edits
/saveddata/<str>           → POST: Save doctor profile edits
/accounts/signup_patient   → Patient registration
/accounts/sign_in_patient  → Patient login
/accounts/signup_doctor    → Doctor registration
/accounts/sign_in_doctor   → Doctor login
/accounts/sign_in_admin    → Admin login
/accounts/logout           → Logout (clears session)
/post                      → POST: Send chat message (returns JSON)
/chat_messages             → GET: Poll for new messages (returns HTML partial)
/post_feedback             → POST: Submit feedback
```

### AJAX Endpoints

Two views return `JsonResponse` (not HTML):
- `POST /checkdisease` — returns `{predicteddisease, confidencescore, consultdoctor}`
- `POST /post` — returns `{msg}` for the chat system

Chat polling uses a GET to `/chat_messages` which returns an HTML partial (`chat_body.html`), suggesting a simple polling strategy (not WebSocket).

### Session Management

Sessions store key state across requests:

```python
request.session['patientusername']   # Set on patient login
request.session['doctorusername']    # Set on doctor login / consultation booking
request.session['doctortype']        # Set after disease prediction (specialist type)
request.session['diseaseinfo_id']    # Set after prediction (links to DB record)
request.session['consultation_id']   # Set after booking / viewing consultation
```

---

## 6. Authentication & Authorization

### Role-Based Access Pattern

Django's built-in `User` model is extended via OneToOne relationships. Role-checking is done manually in views:

```python
# Patient check
if user.patient.is_patient == True:
    auth.login(request, user)

# Doctor check
if user.doctor.is_doctor == True:
    auth.login(request, user)

# Admin check
if user.is_superuser == True:
    auth.login(request, user)
```

The `try/except` blocks catch `RelatedObjectDoesNotExist` — if a user has no patient profile, `user.patient` raises an exception, which is caught and redirects to the login page.

### Known Security Issues (Good to Mention in Interviews)

1. **No `@login_required` decorators** — views check `request.user.is_authenticated` manually, but inconsistently. Some views don't check at all.
2. **Session-based authorization** — views rely on session keys like `patientusername` and `doctorusername` without verifying they match `request.user`, creating potential privilege escalation.
3. **Hardcoded SECRET_KEY** — `'v3v5vfsn0xxjtmb=eoawoiw$5br4g0r&jy_l39995h_93l+-z5'` is committed to the repo.
4. **Database credentials committed** — `.env` file with a PostgreSQL connection string is checked in.
5. **`DEBUG = True` and `DEBUG = False` both set** — `DEBUG = False` at the bottom of settings.py overwrites `DEBUG = True` set earlier, but this creates confusion.

---

## 7. Deployment & DevOps

### Deployment History (from settings.py comments)

The project has been deployed to multiple platforms, each requiring different DB configs:

| Platform | Database | Notes |
|---|---|---|
| Local | PostgreSQL (localhost) | Original dev setup |
| Heroku | Via dj_database_url | WhiteNoise for static files |
| Vercel + Supabase | Supabase PostgreSQL | SSL required |
| Vercel + Neon | Neon DB (ap-southeast-1) | SSL required |
| Render | Render PostgreSQL | Final config in .env |

### Current Active Config

```python
# Uses dj_database_url to read DATABASE_URL from environment
DATABASES = {
    'default': dj_database_url.config(default='sqlite:///db.sqlite3')
}
```

The `.env` file contains:
```
DATABASE_URL=postgresql://predico_db_user:...@...oregon-postgres.render.com/predico_db
```

### Static Files

- **WhiteNoise** middleware serves static files in production (no CDN)
- `STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'`
- Templates directory doubles as static files directory

### Procfile (for Heroku/Render)

```
web: gunicorn disease_prediction.wsgi
```

### vercel.json

Routes all requests through `manage.py` — non-standard Django deployment via Vercel's Python builder.

---

## 8. Code Deep-Dives & Gotchas

### Model Loading — Global Scope

```python
# main_app/views.py — top of file
model = jb.load('trained_model')
```

This loads the model once when the Django application starts, not on every request. This is correct and efficient — reloading a large model per request would be extremely slow. However, it assumes the `trained_model` file is at the working directory root.

### ArrayField — PostgreSQL Dependency

```python
from django.contrib.postgres.fields import ArrayField
symptomsname = ArrayField(models.CharField(max_length=200))
```

This is PostgreSQL-specific. It will fail with SQLite during local dev if you switch DBs. The `dj_database_url` fallback to `sqlite:///db.sqlite3` would break this model.

### The if/elif Bug in Doctor Mapping

```python
if predicted_disease in Rheumatologist:
    consultdoctor = "Rheumatologist"
if predicted_disease in Cardiologist:       # ← should be elif
    consultdoctor = "Cardiologist"
elif predicted_disease in ENT_specialist:   # ← rest are elif
    ...
```

The first two conditions are both `if`, not `if/elif`. A disease in the Rheumatologist list correctly sets `consultdoctor = "Rheumatologist"`, but the subsequent `if predicted_disease in Cardiologist` then runs. If the disease is NOT in Cardiologist, it falls into the `elif` chain. This is likely harmless since the lists are disjoint, but it's a structural bug.

### rating_is Property — N+1 Risk

```python
@property
def rating_is(self):
    new_rating = 0
    rating_obj = rating_review.objects.filter(doctor=self.doctor)
    for i in rating_obj:
        new_rating += i.rating
    new_rating = new_rating / len(rating_obj)
    return int(new_rating)
```

This makes a DB query every time the property is accessed. If rendered in a list of many rating_review objects, it generates N+1 queries. Better: use Django's `Avg` aggregate:
```python
from django.db.models import Avg
rating_review.objects.filter(doctor=self.doctor).aggregate(Avg('rating'))
```

### Chat System — Polling vs WebSocket

The current chat is built with polling (`/chat_messages` returns an HTML partial every few seconds). For a real application, this would be replaced with Django Channels + WebSockets. Worth mentioning as a scalability improvement.

### CSRF Handling

The `@csrf_exempt` decorator is imported in `accounts/views.py` but not actually applied to any view. CSRF protection is active (it's in MIDDLEWARE).

---

## 9. Interview Q&A — Software Developer / Backend Developer

**Q: Walk me through the architecture of Predico.**
A: Predico is a Django monolith with three custom apps — `main_app` for core business logic, `accounts` for authentication, and `chats` for the chat system. The ML model is loaded at server startup using joblib and stored in module scope. The database is PostgreSQL. In production it's deployed on Render with WhiteNoise serving static files and gunicorn as the WSGI server.

**Q: Why use Django's built-in User model instead of a custom User model?**
A: The project extends the built-in User model using OneToOne relationships to `patient` and `doctor` models. This is one of Django's officially recommended patterns. The benefit is that all of Django's auth machinery (authentication, sessions, admin) works out of the box. The trade-off is an extra JOIN on every profile access.

**Q: How does session management work in this app?**
A: After login, the username is stored in `request.session`. Subsequent views retrieve this username to query the database for the user's patient or doctor object. Django sessions are server-side by default (stored in the database), with only a session ID cookie sent to the client.

**Q: What is WhiteNoise and why is it used?**
A: WhiteNoise is a Python library that allows Django to serve its own static files in production without a separate web server like Nginx. It compresses files and adds cache headers. It's added as middleware and configured with `CompressedManifestStaticFilesStorage`. It's suitable for small-to-medium apps but for high traffic, a CDN or Nginx would be preferred.

**Q: What's `dj_database_url` and why is it used?**
A: `dj_database_url` parses a `DATABASE_URL` environment variable (a standard connection string format) into Django's `DATABASES` dictionary format. This is the twelve-factor app approach — configuration in the environment, not in code. It lets you change the database without modifying settings.py.

**Q: How would you add proper authorization (beyond what's in the project)?**
A: Apply Django's `@login_required` decorator to all protected views. Use `@user_passes_test` or custom decorators to enforce role checks (patient vs. doctor vs. admin). Replace manual `request.session['patientusername']` lookups with `request.user.patient` directly, verifying that `request.user` matches the expected role.

**Q: How does the consultation chat work?**
A: When a patient books a consultation, a `consultation` record is created linking the patient, doctor, and diseaseinfo objects. The consultation ID is stored in the session. Chat messages are `Chat` objects with a ForeignKey to the consultation. The frontend polls `/chat_messages` periodically, which queries all messages for the session's consultation ID and returns an HTML partial that replaces the chat body.

---

## 10. Interview Q&A — Data Scientist / ML Engineer

**Q: Why Random Forest for disease prediction?**
A: Random Forest is well-suited here because the features are binary (symptom present/absent), the target has 41 classes, and the dataset is relatively small. RF is robust to overfitting, requires no feature scaling, handles multi-class classification natively, and provides `predict_proba()` for confidence scores. It also has feature importance built-in for interpretability.

**Q: What is SMOTE and why was it used?**
A: SMOTE (Synthetic Minority Oversampling Technique) generates synthetic training samples for minority classes by interpolating between existing minority-class samples in feature space. It was used because some diseases in the dataset have fewer examples than others, which would cause the model to be biased toward more common conditions. Crucially, SMOTE is applied before the train/test split so synthetic samples don't leak into the test set.

**Q: How does `predict_proba()` work for Random Forest?**
A: Each tree in the forest votes for a class. `predict_proba()` returns the fraction of trees that voted for each class. The maximum value across all classes is taken as the confidence score. For a well-calibrated Random Forest, this provides a reasonable probability estimate.

**Q: How is the feature vector constructed for inference?**
A: A list of 132 zeros is created — one element per symptom in the fixed `symptomslist`. For each symptom the patient selected, the corresponding index is set to 1. This creates a binary vector matching the training data format. Maintaining the exact order of symptoms is critical — a mismatch would silently produce wrong predictions.

**Q: What are the limitations of this ML approach?**
A: The model maps symptoms to diseases using static binary features. It can't capture symptom severity, duration, or patient history. It was trained on a curated dataset that may not reflect real-world symptom distributions. It also doesn't account for co-morbidities. The model is not updated online — it's static after training. For production medical use, continuous retraining with new data and human oversight would be essential.

**Q: How would you evaluate this model beyond accuracy?**
A: For a multi-class medical classification problem: precision and recall per class (a missed cancer diagnosis is far worse than a false positive), F1-score, confusion matrix to spot systemic misclassifications, AUC-ROC for each class using OvR strategy, and calibration curves to verify that predicted probabilities match actual frequencies.

**Q: What hyperparameters were tuned and how?**
A: `n_estimators` (50, 100, 150) and `max_depth` (None, 10, 20, 30) using `GridSearchCV` with 3-fold `StratifiedKFold` cross-validation. Stratified k-fold ensures each fold has a proportional representation of each class — important for imbalanced datasets.

---

## 11. Interview Q&A — Full Stack Developer

**Q: How does the symptom selection and disease prediction flow work end-to-end?**
A: The patient UI renders a form with checkboxes for 132 symptoms (sorted alphabetically). On form submission, JavaScript collects checked symptom names into an array and makes a POST AJAX request to `/checkdisease` with `symptoms[]`. Django receives this with `request.POST.getlist("symptoms[]")`, builds the feature vector, runs inference, saves the result to the DB, stores the diseaseinfo ID in the session, and returns a JSON response. The frontend renders the predicted disease, confidence score, and specialist recommendation dynamically without a page reload.

**Q: Why are templates stored in the templates directory rather than per-app?**
A: The `TEMPLATES` settings configure `DIRS` to look in `os.path.join(BASE_DIR, 'templates')`, meaning all templates live in one top-level directory organized by role (patient/, doctor/, admin/, consultation/). This is a valid Django approach for projects where templates are a single frontend team's concern. Per-app templates work better when apps are meant to be reusable across projects.

**Q: How does the dual-role login system work in the frontend?**
A: There are separate login pages, separate signup forms, and separate URLs for patients vs. doctors vs. admins. Each login view checks the appropriate role flag on the related model (`user.patient.is_patient` or `user.doctor.is_doctor`). After login, the user is redirected to the appropriate dashboard. There is no single login page with role detection.

**Q: How would you add real-time chat instead of polling?**
A: Replace the polling-based chat with Django Channels, which adds WebSocket support to Django. Install `channels`, configure an ASGI server (e.g., Daphne or Uvicorn), create a `ChatConsumer` that handles WebSocket connect/disconnect/receive, use Redis as the channel layer, and have the frontend use the WebSocket API instead of periodic GET requests.

**Q: What is WhiteNoise's role and what would you use at scale?**
A: WhiteNoise lets Django serve static files directly. For higher traffic, you'd use a CDN (Cloudflare, AWS CloudFront) to cache and serve static files globally, and an object storage service (AWS S3) for user-uploaded media. An Nginx reverse proxy in front of gunicorn is also standard for production.

---

## 12. Interview Q&A — System Design

**Q: How would you redesign this app to support 100,000 users?**
A: Separate the ML inference service from the Django app — deploy it as a microservice (Flask or FastAPI) with its own horizontal scaling. Use Redis for session storage and caching. Replace polling chat with WebSockets (Django Channels + Redis channel layer). Use PostgreSQL read replicas for heavy read traffic. Add a CDN for static assets. Consider a task queue (Celery + Redis) for async operations like sending notification emails.

**Q: How would you make the disease prediction more accurate over time?**
A: Implement a feedback loop — track whether a predicted disease turned out to be correct (via doctor confirmation in the consultation). Store prediction outcomes. Retrain the model periodically with augmented data. Consider using a more powerful model (XGBoost, LightGBM, or even fine-tuned transformer on clinical text). Add continuous monitoring for model drift.

**Q: What database indexes would you add?**
A: Index `consultation.patient_id` and `consultation.doctor_id` for consultation history queries. Index `rating_review.doctor_id` for the rating aggregation query. Index `Chat.consultation_id` for chat message retrieval. Since `patient` and `doctor` use `user_id` as the primary key (OneToOne with PK), those are already indexed.

**Q: How would you handle the `trained_model` file in a container/cloud environment?**
A: In a containerized deployment, the model file would be baked into the Docker image. For a multi-instance deployment, the model should be versioned and stored in object storage (S3) and loaded at startup, or better, served by a dedicated ML serving layer (TorchServe, TensorFlow Serving, or a FastAPI microservice).

**Q: How would you improve the doctor recommendation beyond keyword matching?**
A: Build a proper recommendation system — collaborative filtering based on patient ratings, content-based filtering using doctor specialization + consultation history, or a hybrid approach. Factor in doctor availability, patient location, and rating. Add a verification step for doctor credentials using a third-party medical council API.

---

## 13. Known Issues & How You'd Fix Them

| Issue | Location | Fix |
|---|---|---|
| Hardcoded SECRET_KEY | settings.py | Use `os.environ.get('SECRET_KEY')` |
| DB credentials in .env committed to repo | .env | Add `.env` to `.gitignore`, use environment variables in CI/CD |
| No `@login_required` on protected views | All views | Add decorator, use middleware or CBVs with `LoginRequiredMixin` |
| Session username not validated against `request.user` | main_app/views.py | Use `request.user.patient` directly |
| `if/elif` bug in specialist mapping | main_app/views.py line ~170 | Change first `if Cardiologist` to `elif` |
| N+1 query in `rating_is` property | main_app/models.py | Use `Avg` aggregate from `django.db.models` |
| No input validation on symptom selection | checkdisease view | Validate that `psymptoms` values exist in `symptomslist` |
| No rate limiting on prediction endpoint | /checkdisease | Add Django rate limiting or `django-ratelimit` |
| Static model file path is relative | main_app/views.py | Use `os.path.join(settings.BASE_DIR, 'trained_model')` |
| `DEBUG = True` and `DEBUG = False` both set | settings.py | Clean up settings.py, use environment-specific settings files |
| ArrayField makes SQLite impossible | main_app/models.py | Use JSONField (Django 3.1+) for cross-DB compatibility |
| Chat uses polling, not WebSockets | chats/, main_app/ | Migrate to Django Channels |
| Passwords compared with `==` but no strength check beyond validators | accounts/views.py | Already uses Django validators; add frontend strength meter |
| No email verification on signup | accounts/views.py | Add email confirmation flow |

---

## 14. Improvements & Extensions to Mention

### Technical Improvements

- **REST API layer** — Build a DRF (Django REST Framework) API to decouple frontend from backend and enable mobile apps
- **Celery + Redis** — Async tasks: send email notifications when a doctor responds, generate PDF consultation reports
- **Model versioning** — Use MLflow or DVC to track model experiments and versions
- **CI/CD pipeline** — GitHub Actions for automated testing, linting (flake8, black), and deployment
- **Containerization** — Dockerize the app with docker-compose for local dev (Django + PostgreSQL + Redis)
- **Comprehensive test suite** — Unit tests for ML inference, integration tests for auth flows, API tests for prediction endpoint

### Feature Extensions

- **Symptom autocomplete** — Replace 132 checkboxes with a searchable autocomplete input
- **Multi-language support** — i18n for Hindi and regional Indian languages
- **Prescription system** — Doctors can upload/send prescriptions as files
- **Video consultation** — Integrate WebRTC for video calls within the consultation
- **Appointment scheduling** — Time-slot based booking instead of instant consultation
- **Notification system** — Email/SMS notifications via SendGrid/Twilio
- **Analytics dashboard** — Admin view with prediction trends, most common diseases, active consultations

### ML Improvements

- **Symptom severity weighting** — Instead of binary features, use 0/1/2/3 for absent/mild/moderate/severe
- **Differential diagnosis** — Return top-3 predicted diseases with probabilities, not just the most likely one
- **Explainability** — Use SHAP values to show which symptoms most influenced the prediction
- **Online learning** — Retrain model periodically with confirmed diagnosis data from consultations
- **NLP symptom input** — Allow patients to describe symptoms in natural language; use NLP to extract structured features

---

## Quick Reference Card for Interviews

| Topic | Key Point |
|---|---|
| Framework | Django 2.2.x (Python) |
| ML Model | Random Forest Classifier via scikit-learn |
| Database | PostgreSQL (via psycopg2 + dj_database_url) |
| Model Serialization | joblib (`jb.load('trained_model')`) |
| Class Imbalance | SMOTE from imbalanced-learn |
| Hyperparameter Tuning | GridSearchCV + StratifiedKFold (3-fold) |
| Features | 132 binary symptom features |
| Target Classes | 41 diseases |
| Static Files | WhiteNoise middleware |
| Deployment | Render (Procfile + gunicorn) |
| Auth | Django built-in + OneToOne role models |
| Chat | HTTP polling (not WebSocket) |
| Key AJAX Endpoints | `/checkdisease` (POST), `/post` (POST), `/chat_messages` (GET) |
| Session Keys | `patientusername`, `doctorusername`, `diseaseinfo_id`, `consultation_id` |
| Notable Bug | `if/elif` mismatch in specialist mapping |
| Notable Design Issue | N+1 in `rating_is` property |
| PostgreSQL-specific | `ArrayField` for symptom storage |
