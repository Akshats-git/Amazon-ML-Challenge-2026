# Amazon ML Challenge 2026 — Complete Prep Notes

**Source:** AWS Builder Center post by Jatin (Developer Advocate @ AWS)
**Published:** Sep 21, 2026 · **Last modified:** Sep 23, 2026
**Session recording:** Twitch — Sep 21, 5:00–6:00 PM IST

---

## TL;DR

- 79,000+ students registered
- $200 in free AWS credits for every participant
- 72-hour ML hackathon
- Prizes worth INR 2.25 lakh + PPIs at Amazon
- A prep/best-practices session was hosted on Sep 21 (recording on Twitch)

---

## 1. What is the Amazon ML Challenge?

A **two-stage machine learning competition** open only to engineering students across India. Teams get a real-world Amazon dataset and build an ML model to solve a business problem.

### Eligibility

| Criterion | Requirement |
|---|---|
| Course | PhD / M.E. / M.Tech / M.S. / B.E. / B.Tech at any engineering campus in India |
| Graduation year | 2027 or 2028 |
| Team size | 2–4 members (**cross-college teams allowed**) |
| Mandatory | A valid **AWS Builder Center Profile ID** |

### Rewards

| Reward | Who gets it |
|---|---|
| INR 1,00,000 + certificates + goodies | 🥇 Winners |
| INR 75,000 + certificates + goodies | 🥈 First Runners-up |
| INR 50,000 + certificates + goodies | 🥉 Second Runners-up |
| Pre-Placement Interviews at Amazon | 🎯 Top 50 teams |
| $200 AWS Credits | 💰 All participants |
| Extra $100 credits (at the 48-hour mark) | ⚡ Top 500 teams |
| Certificates + Amazon SWAG | 🏅 Top 10 teams + Top 10 women-only teams |

### Timeline

| Phase | Dates |
|---|---|
| 📋 Registration | 7 – 22 Sep 2026 |
| 🔴 Prep virtual session | 21 Sep 2026, 5:00–6:00 PM IST (recording available) |
| ⚔️ 72-hour hackathon | 25 – 27 Sep 2026 |
| 📊 Top 50 results | 2 Oct 2026 |
| 🏆 Grand Finale | 7 Oct 2026 — Top 10 present to Amazon Scientists |

---

## 2. AWS Builder Center — Free Dev Playground

A Builder Center profile is **mandatory for registration**, but it is useful beyond being a checkbox.

| Feature | Why it matters |
|---|---|
| 🧪 Free sandbox environments | Real AWS accounts, no credit card. 8 hours per week. |
| 📚 Hands-on workshops | Level 100 to 400. Real services, not just videos. |
| 🌐 Public builder profile | Portfolio URL for LinkedIn and résumés |
| 👥 Student Builder Groups | University communities in 60+ countries |

**Setup:** go to `builder.aws.com` → Join → verify email → pick an alias. Under 5 minutes, no credit card.

---

## 3. Student Rewards — Up to $579 in Free Value

Launched by AWS on **August 20, 2026**. Credit card is **never** required.

```
✅ Verify your student status
   └──→ 12 months Skill Builder Premium ($449 value)

🏅 Earn 7 badges on Builder Center
   └──→ $10 AWS Credits

🏅🏅 Earn 14 badges
   └──→ $20 AWS Credits

🏅🏅🏅 Earn 21 badges
   └──→ $100 AWS Certification Exam Voucher

💰 Total Value: $579
💳 Credit Card Required: NEVER
```

Badges come from simple daily activity — signing in, commenting, publishing articles, submitting feature requests. Most students can reach 7 badges in the first week.

> **Pro tip:** Start earning badges now. The $10 and $20 credits arrive early and can be spent on SageMaker during the challenge.

---

## 4. AWS Free Tier — Starting at $0

Every participant gets **$200 in AWS credits** just for registering.

**How the $200 breaks down:** $100 arrives on sign-up, and completing 5 starter activities (creating an EC2 instance, an RDS database, etc.) unlocks another $100.

Even without credits, new accounts get a generous free tier:

| Service | Free tier |
|---|---|
| SageMaker Notebooks | 250 hours on `ml.t3.medium` (2 months) |
| SageMaker Training | 50 hours on `ml.m5.xlarge` (2 months) |
| SageMaker Inference | 125 hours on `ml.m5.xlarge` (2 months) |
| S3 Storage | 5 GB always free |
| Lambda | 1M requests/month always free |

### Caveats

- Set **billing alerts** in the console immediately after creating your account.
- Always **delete endpoints** when done — they charge roughly $0.12/hour.
- Use the **`us-east-1`** region for best compatibility.

---

## 5. Cloud Fundamentals — Core Services to Know

| Service | What it does | Why you need it |
|---|---|---|
| Amazon EC2 | Virtual servers in the cloud | SageMaker training jobs run on EC2 instances behind the scenes |
| Amazon S3 | Object storage (files, data, models) | Data goes in, trained models come out — everything flows through S3 |
| Amazon DynamoDB | NoSQL database | Store results, metadata, or feature stores at scale |
| IAM | Permissions and access control | The "role" that lets SageMaker reach your data and spin up machines |
| Amazon VPC | Your private network inside AWS | All SageMaker resources run inside a VPC — like your own private WiFi, keeping things secure and isolated. Quick Setup creates one automatically |

You don't need to master all of these, but knowing what they do and how they connect makes SageMaker click much faster.

---

## 6. Stream vs Blog — Important Differences

The live stream used SageMaker's **managed infrastructure**; the blog does everything **locally**. Both work.

| Topic | Managed approach | Local approach |
|---|---|---|
| **Studio vs Notebook Instance** | Studio is a full workspace with multiple apps, but needs a Domain setup (can take time) | A Notebook Instance is a plain Jupyter notebook — no Domain, no hassle. Same code runs on both. |
| **Training Job vs local training** | A Training Job spins up a separate, more powerful machine that pulls data from S3. Use it for big machines or GPUs. | Local training runs inside the notebook on its own CPU. Data is already in memory, no S3 upload. Use when data is small. |
| **Endpoint vs local prediction** | An Endpoint is a 24/7 hosted API for production apps — costs ~$0.12/hour even when idle | `model.predict()` returns instantly inside the notebook |

**Recommended for the ML Challenge:** Notebook Instance + local training + local prediction. You submit a CSV, not a running API.

> Notes from the author: the stream used SageMaker Studio; brand-new Free Tier accounts may have to wait while things provision behind the scenes, so a Notebook Instance is the safer choice for a timed challenge. Long-standing AWS accounts can follow the stream version fine. The stream trains on a dedicated SageMaker training instance, so its code differs slightly (S3 upload, endpoint deployment) — both versions work.

---

## 7. The Demo — Build, Train and Predict on SageMaker

A ~30-minute hands-on walkthrough from raw data to live predictions.

### Step 0 — Create a SageMaker Notebook Instance

Using a Notebook Instance (not Studio) avoids quota issues on all accounts.

1. Open the **SageMaker AI console**
2. Left sidebar → **Applications and IDEs → Notebook → Notebook instances**
3. Click **Create notebook instance**
4. Set a **Name** (demo used `ml-challenge-notebook`)
5. Set **Instance type** to `ml.t3.medium` (free tier — 250 hours)
6. Under **IAM Role**, select **Create a new role** → leave defaults → **Create role**
7. Click **Create notebook instance**
8. Wait 2–3 minutes for status to become **InService**
9. Click **Open JupyterLab**
10. Click **+** → select **conda_python 3** notebook

### Step 1 — Install libraries

```python
!pip install xgboost scikit-learn -q
```

Installs XGBoost (the ML algorithm) and scikit-learn (metrics and data splitting). `-q` keeps output clean.

> Nothing may appear to happen — watch the bottom status line to see whether the kernel is busy.

### Step 2 — Set up the SageMaker session

```python
import sagemaker
import boto3
import pandas as pd
import numpy as np
import time

session = sagemaker.Session()
role = sagemaker.get_execution_role()
region = session.boto_region_name
bucket = session.default_bucket()

print(f"Region: {region}")
print(f"Role: {role}")
print(f"Bucket: {bucket}")
```

- `Session()` connects to SageMaker
- `get_execution_role()` fetches the IAM permission "pass" that lets SageMaker reach your data
- `default_bucket()` is your auto-created S3 bucket

Three lines and the whole infrastructure is ready.

### Step 3 — Load the dataset

```python
data = pd.read_csv(
    f"s3://sagemaker-example-files-prod-{region}/datasets/tabular/synthetic/churn.txt"
)
df = data.copy()

print(f"Dataset shape: {df.shape}")
df.head()
```

The dataset is a **synthetic telecom customer dataset** — roughly 5,000 customers with features like call minutes, charges and customer-service calls. Not real customer data; AWS generated it for learning. AWS hosts public datasets in S3 for tutorials, so it's read directly from their public bucket with no upload.

For the ML Challenge you'll upload your own dataset to **your** S3 bucket and swap in that URL — and give your notebook's IAM role permission to read it.

### Step 4 — Explore the data

```python
print("Target variable distribution:")
print(df["Churn?"].value_counts())
print(f"\nChurn rate: {df['Churn?'].value_counts(normalize=True)['True.']:.1%}")
```

```python
print(f"Features: {df.shape[1]} columns, {df.shape[0]} rows")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"\nColumn types:\n{df.dtypes.value_counts()}")
```

The target column is `Churn?` — `True` means the customer left, `False` means they stayed. This dataset is roughly a 50/50 split.

**Always check target distribution first.** It tells you what kind of problem you have. Here it's balanced; ML Challenge data may be heavily skewed, which changes how you build and evaluate the model.

21 columns, zero missing values — a clean dataset. Challenge data won't be this clean: expect missing values and noise. But the exploratory step is always the same — understand the data before modelling it.

### Step 5 — Feature engineering

```python
from sklearn.model_selection import train_test_split

# Drop non-predictive columns
df = df.drop("Phone", axis=1)

# Area Code is a code, not a number
df["Area Code"] = df["Area Code"].astype(object)

# Remove redundant charge columns (charges = minutes x rate)
df = df.drop(["Day Charge", "Eve Charge", "Night Charge", "Intl Charge"], axis=1)

# One-hot encode categorical features (Yes/No -> 1/0)
model_data = pd.get_dummies(df)

# Move target to first column (XGBoost convention)
model_data = pd.concat(
    [model_data["Churn?_True."],
     model_data.drop(["Churn?_False.", "Churn?_True."], axis=1)],
    axis=1
)
model_data = model_data.astype(float)

print(f"Processed: {model_data.shape}")
model_data.head()
```

**Why drop `Phone`?** It's a unique ID — every customer has a different one, so it predicts nothing. Drop unique IDs, serial numbers and row numbers.

**Why move the target to column 0?** SageMaker's built-in XGBoost expects a CSV with **no column headers** and the **target in the first column**. This is the number-one beginner mistake — leaving headers in, or burying the target mid-table, makes the model crash or train on the wrong thing.

> Two rules: **no headers, target first.** (And how do you learn rules like these? Read the documentation.)

> **Key takeaway:** Feature engineering often matters more than algorithm choice. Spend ~70% of your time here.

### Step 6 — Split the data

```python
# Split into train (67%), validation (22%), test (11%)
train_data, validation_data = train_test_split(model_data, test_size=0.33, random_state=42)
validation_data, test_data = train_test_split(validation_data, test_size=0.33, random_state=42)

# Separate test labels (we'll use these to check accuracy later)
test_target = test_data['Churn?_True.']
test_data_no_target = test_data.drop(['Churn?_True.'], axis=1)

# Separate features and labels for train/validation
train_features = train_data.iloc[:, 1:]
train_labels = train_data.iloc[:, 0]
val_features = validation_data.iloc[:, 1:]
val_labels = validation_data.iloc[:, 0]

print(f"Training:   {train_data.shape[0]} rows")
print(f"Validation: {validation_data.shape[0]} rows")
print(f"Test:       {test_data.shape[0]} rows")
```

**Why three sets?**
- **Training** = the textbook the model learns from
- **Validation** = practice tests to check progress
- **Test** = the final exam, never seen until the end

**Golden rule:** never touch the test data until you're finished experimenting.

### Step 7 — Train with XGBoost (locally)

```python
# Create XGBoost's special data format
dtrain = xgb.DMatrix(train_features, label=train_labels)
dval = xgb.DMatrix(val_features, label=val_labels)

# Hyperparameters
params = {
    "max_depth": 5,
    "eta": 0.2,
    "gamma": 4,
    "min_child_weight": 6,
    "subsample": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss"
}

print("Training locally... (no separate machine needed)")
model = xgb.train(
    params, dtrain, num_boost_round=100,
    evals=[(dtrain, "train"), (dval, "validation")],
    verbose_eval=10
)
print("\nTraining complete!")
```

Runs inside the notebook using the open-source XGBoost library — no SageMaker training job, no separate cloud machine, no quotas. For the ML Challenge dataset, this is all you need.

### Step 8 — Get predictions

```python
# Predict on test data
dtest = xgb.DMatrix(test_data_no_target)
predictions = model.predict(dtest)

# Show sample predictions
print("PREDICTIONS (probability of churn):")
print("-" * 55)
for i in range(10):
    pred = predictions[i]
    actual = test_target.iloc[i]
    predicted = "CHURN" if pred > 0.5 else "STAY"
    actual_lbl = "CHURN" if actual == 1.0 else "STAY"
    status = "CORRECT" if predicted == actual_lbl else "WRONG"
    print(f"  Customer {i+1}: {pred:.3f} -> {predicted:5s} (Actual: {actual_lbl:5s}) {status}")
```

Each prediction is a probability between 0 and 1. Above 0.5 = CHURN, below = STAY. Confidence matters: 0.92 is very confident, 0.51 is barely sure.

### Step 9 — Evaluate the model

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

binary_preds = (predictions > 0.5).astype(int)

print("FULL TEST SET RESULTS:")
print(f"  Accuracy:  {accuracy_score(test_target, binary_preds):.1%}")
print(f"  Precision: {precision_score(test_target, binary_preds):.1%}")
print(f"  Recall:    {recall_score(test_target, binary_preds):.1%}")
print(f"  F1 Score:  {f1_score(test_target, binary_preds):.1%}")
```

**Result:** ~92% accuracy, zero hyperparameter tuning, trained in seconds — that's XGBoost on tabular data.

### Step 10 — Save your model (optional)

```python
# Save the trained model locally
model.save_model("xgboost_churn_model.json")
print("Model saved! You can reload it anytime with:")
print("  loaded_model = xgb.Booster()")
print("  loaded_model.load_model('xgboost_churn_model.json')")
```

For the ML Challenge, save your model and predictions — you submit a zip file with your code plus an approach document.

### Cleanup

When done for the day: SageMaker console → **Notebook instances** → select your notebook → **Stop**. That halts billing and you can restart later with files intact.

Unlike endpoints, notebook instances cost nothing while stopped. **Stop, don't delete** — deleting removes your files. After the challenge, delete the instance.

---

## 8. Resources

- 📺 Live session recording (Sep 21, 5–6 PM IST) — Twitch
- AWS Builder Center — `builder.aws.com`
- Student Rewards signup — `aws.amazon.com/builder`
- ML Challenge registration — Unstop
- SageMaker free tier / pricing — `aws.amazon.com/sagemaker/pricing`
- SageMaker SDK v3 tutorial — `sagemaker.readthedocs.io`
