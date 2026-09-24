# Amazon ML Challenge 2026: AWS Builder Center & AWS Free Tier

## Instructions

---

## AWS Builder ID and sign-in

### What is AWS Builder Center and how do I access it?

AWS Builder Center is the go-to site for builders to connect with the AWS community. It provides a seamless experience for discovering community-published articles, networking with like-minded professionals, and submitting product improvement ideas through the Wishlist. You can browse articles and search for builders without signing in. However, signing in with your Builder ID unlocks even more features!

After signing in, you can follow other builders and receive notifications, engage with content by liking, commenting, or reporting, and access Wishlist to view, upvote, and comment on product suggestions.

### What is AWS Builder ID and how does it work with AWS Builder Center?

AWS Builder ID is your personal credential for learning, connecting, and experimenting with AWS outside of the AWS Management Console. It's separate from your AWS account or AWS Partner Network credential. If you already use Builder ID for AWS Skill Builder, or AWS re:Post, you can use the same Builder ID for AWS Builder Center. This single credential allows you to create a Builder Center profile, publish articles, create wishes, engage with content across multiple AWS communities, and more.

### How do I get started with Builder ID?

To check, visit the AWS Builder Center and click **Sign In** in the top navigation. Select **Already have a Builder ID? Sign in** and enter your email address. If an account exists, you'll be prompted for a password. If not, you'll be guided through the creation process. If you've previously used AWS Skill Builder, or AWS re:Post, you likely already have a Builder ID.

**For new Builder ID creation:**

1. On the AWS Builder Center top navigation, click **Sign in**.
2. On the **Create AWS Builder ID** page, enter **Your email address**.
3. Choose **Next**.
4. Enter **Your name**, and then choose **Next**.
5. On the **Email verification** page, enter the verification code that we sent to your email address. Choose **Verify**. Depending on your email provider, it might take a few minutes for you to receive the email. Check your spam and junk folders for the code. If you don't see the email from AWS after five minutes, choose **Resend code**.
6. After we verify your email, on the **Choose a password** page, enter a **Password** and **Confirm password**.
7. If a Captcha appears as additional security, enter the characters that you see.
8. Choose **Create AWS Builder ID**.

**Once you have a Builder ID, signing in to AWS Builder Center is straightforward:**

1. On the AWS Builder Center top navigation, click **Sign in**.
2. Enter your Builder ID email and password.
3. If required, complete any additional verification steps.
4. For first-time builders, you'll be prompted to choose an alias to complete your Builder Center profile.

Remember, your Builder ID is your key to accessing various AWS community programs. It allows you to create a Builder Center profile, publish articles, submit wishes, and engage with content across different AWS learning and community services. If you encounter any issues during sign-up or sign-in, AWS Builder Support is available to assist you.

### Can I sign into AWS Builder Center with my AWS Account credential?

No. Your AWS Builder ID is separate from any AWS account or AWS Management Console.

### How do I create an AWS Builder Center alias?

Your AWS Builder Center alias is a unique identifier for your profile on builder.aws.com and is what makes your profile URL unique (`builder.aws.com/community/@youralias`). You are asked to set your alias the first time signing into Builder Center. If you created an alias prior to Builder Center, this will also be your Builder Center alias.

Your Builder Center alias:

- Must be between 3 and 19 characters in length
- Must begin with a letter and can include lowercase letters (a-z) and numbers (0-9)
- Cannot include spaces or non-alphanumeric characters (`~!@#$%^&*_-+=\`|\(){}[]:;"'<>,.?/`)
- Cannot violate the AWS Builder Terms

### Where do I find my Builder Center alias?

After you sign in to AWS Builder Center, click your name in top navigation. Then click **Manage Profile**. You'll see your alias on this page.

---

## Student Verification and Rewards

### What is Builder Center Student Rewards, and what do verified students receive?

Student Rewards is a free program that gives verified students the tools to learn AWS, build real projects, and earn a certification at no cost. Once you verify your student status on AWS Builder Center and complete your profile, you receive a welcome package, then earn additional rewards as you progress through Builder Center's badge system.

- **Welcome package:** a 12-month AWS Skill Builder subscription with full access to digital courses, game-based learning, labs, and exam prep content. Earned by completing student verification and your Builder Center profile.
- **First milestone:** $10 in AWS Credits. Earned by collecting 7 Builder Center badges.
- **Second milestone:** $20 in AWS Credits. Earned by collecting 14 Builder Center badges.
- **Certification voucher:** a $100 AWS Certification exam voucher, redeemable toward AWS Certified Cloud Practitioner. Earned by collecting all 21 Builder Center badges.

You can get started on the Student Rewards page.

### How do I get verified, and how long does it take?

Verification takes three steps, and most students finish in a few minutes.

1. **Step 1: Sign in or sign up.** Go to Builder Center and select **Sign in**. An AWS Builder ID can be created for free with an email address, or with a Google, Apple, GitHub, or Amazon account. We recommend using a personal email address, which can be a school address, so that access to your Builder ID and any earned rewards continues after graduation.
2. **Step 2: Verify your student status.** Go to **Edit profile > Student details**, or use the **Student verification** link, then select **Start verification** and follow the steps provided by SheerID, the third-party verification service. You provide your institution name and enrollment details, and upload documentation if prompted.
3. **Step 3: Complete your Builder Center profile.** Upload a profile photo and complete the About section. Both are required to receive the welcome package. These two actions also earn your first two badges, **Photo Finisher** and **Hello, World!**, which starts you on the path to the 7-badge milestone.

Once all three steps are complete, your verified student status is reflected on your Builder Center profile and the 12-month Skill Builder subscription appears in your rewards dashboard within minutes. If SheerID requests additional documentation, typically a class schedule or enrollment letter, verification usually completes within 24 to 48 hours.

### What happens if verification is rejected?

You can try again with alternate documentation. The most common reasons for rejection are that the institution is not eligible, the document submitted was unclear, or the enrollment dates do not reflect current enrollment. If you continue to have trouble, contact SheerID support.

Have questions? Find answers in our detailed FAQ page here.

---

## AWS Free Tier

### What is the AWS Free Tier program?

AWS Free Tier program allows new customers to explore AWS services at no cost for up to 6 months. New customers receive up to **$200 in AWS credits** — $100 upon sign-up and up to $100 more as you explore foundational AWS services. You can access over 30 services with always free offers. Services with an Always Free offer allow you to use the product for free up to specified limits as long as you are an AWS customer.

### How do I get started with AWS Free Tier?

You can visit the AWS Management Console or AWS Free Tier and click on **"Create an AWS Account"** to start the sign up process. During sign-up, you'll choose between two plans — **Free** and **Paid**. With both plans, you can receive up to $200 in AWS credits and access over 30 always-free services that offer free monthly usage. [Ref: Managing Payments in India]

### Free Tier for Builders

Learn how your AWS Free Tier account works, configure your tools, and ship your first deployment. Then keep going: build a serverless API, connect a cloud database, launch a web server, deploy an AI agent. The guides show real code, real output, and what to check when something fails.

- Everything you need to know to get started with AWS Free Tier (Article)
- Setting up your AWS developer environment (Article)

---

## Optimising AWS Free Tier while participating in Amazon ML Challenge

Each team member gets their own AWS Free Tier account. You can pool credits by sharing trained model artifacts across accounts via S3.

### TL;DR

1. Member A trains model, saves artifact to S3
2. Member A adds a bucket policy granting Member B's account read access
3. Member B copies the artifact to their own account, continues iterating
4. Rinse and repeat until you're happy with the output

New AWS accounts receive up to $200 in credits. Within that:

- **SageMaker (2-month trial):** 250 hrs ml.t3.medium notebook + 50 hrs m4.xlarge or m5.xlarge training
- **S3:** 5 GB storage, 20,000 GET requests, 2,000 PUT requests/month

### Option 1: S3 Cross-Account Sharing

**Step 1 — Exchange Account IDs**

AWS Console > top-right account dropdown > copy 12-digit Account ID > share with teammate.

**Step 2 — Member A: Upload model artifact**

```bash
aws s3 mb s3://hackathon-<team-name>
aws s3 cp ./model.tar.gz s3://hackathon-<team-name>/models/
```

**Step 3 — Member A: Grant cross-account access**

S3 Console > Bucket > Permissions > Bucket Policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {
      "AWS": "arn:aws:iam::<TEAMMATE_ACCOUNT_ID>:root"
    },
    "Action": ["s3:GetObject", "s3:ListBucket"],
    "Resource": [
      "arn:aws:s3:::hackathon-<team-name>",
      "arn:aws:s3:::hackathon-<team-name>/*"
    ]
  }]
}
```

**Step 4 — Member B: Pull and continue**

```bash
aws s3 cp s3://hackathon-<team-name>/models/model.tar.gz s3://my-bucket/models/
```

Point your SageMaker job to `s3://my-bucket/models/model.tar.gz` and keep going.

### Option 2: SageMaker Model Export + S3 Transfer

If you're using SageMaker for training, model artifacts are automatically stored in S3. You just need to find and share them.

**Step 1 — Member A: Locate the model artifact**

SageMaker saves training output to:

```
s3://sagemaker-<region>-<account-id>/output/<training-job-name>/output/model.tar.gz
```

Or find it in SageMaker Console > Training Jobs > your job > Output > S3 model artifact.

**Step 2 — Member A: Copy to shared bucket**

```bash
# Copy from SageMaker default bucket to your shared bucket
aws s3 cp \
  s3://sagemaker-<region>-<account-id>/output/<job-name>/output/model.tar.gz \
  s3://hackathon-<team-name>/models/
```

Use the same bucket policy from Option 1 Step 3 to grant teammate access.

**Step 3 — Member B: Import into their SageMaker environment**

```python
from sagemaker.model import Model

model = Model(
    model_data="s3://my-bucket/models/model.tar.gz",
    role=role,
    image_uri="<framework-container-uri>"  # same container used for training
)
```

Member B can now deploy, run inference, or use this as a starting point for further training with their own free tier hours.

### Lazy Alternative — Presigned URLs

Skip all IAM config. Member A runs:

```bash
aws s3 presign s3://hackathon-<team-name>/models/model.tar.gz --expires-in 86400
```

Share the URL. Teammate downloads. Done.

### Monitor Credits

AWS Console > Billing > Free Tier. Hand off to your teammate before you hit limits.

### Caveats / Key Limitations

- **SageMaker Training Jobs are not shareable cross-account** — only the output artifacts (model files) can be transferred via S3. You cannot "resume" a training job in another account.
- **SageMaker Endpoints can't be shared cross-account without AWS Organizations** — not a problem here since you only need to submit a CSV, not host an endpoint.
- **S3 default encryption keys don't work cross-account** — if you used a custom KMS key for encryption, the teammate's account won't be able to decrypt. Stick to default S3 encryption or no encryption.
- **Free plan is time-bound** — the Free plan auto-closes after 6 months or when $200 credits are exhausted. SageMaker's 2-month trial window is shorter than the overall account window.
- **Model compatibility** — both team members must use the same framework version and container image. A model trained on PyTorch 2.0 won't load cleanly in a PyTorch 1.x container.
- **S3 bucket names are globally unique** — if `hackathon-<team-name>` is taken, pick another name.
- **Data transfer costs** — cross-region S3 copies incur data transfer charges (deducted from credits). Keep both accounts in the same region to avoid this.

### Reference Docs

- SageMaker AI Pricing & Free Tier
- S3 cross-account access
- Bucket owner granting cross-account permissions
- Cross-account upload access
- Deploy SageMaker model to different account
- SageMaker cross-account S3 permissions
- Cross-account MLOps with SageMaker Model Registry (blog)
- Cross-account S3 access for SageMaker notebooks (blog)

---

## How do I redeem AWS Credit Codes?

*(For top 500 teams on the leaderboard during the challenge for extra credits)*

AWS credits are promotional codes that apply dollar-value discounts to your AWS bill. They cover most AWS services, allowing you to experiment with cloud infrastructure, deploy applications, and learn AWS technologies with reduced financial risk. The full list of AWS services covered by these promotional credits is available under the 'credits' tab in your AWS account dashboard.

### Redemption Steps

1. **Sign in to the AWS Management Console.**
   - Navigate to the AWS Management Console and sign in with your account credentials.
2. **Access the Billing Dashboard.**
   - Click on your account name in the top-right corner and select "Billing and Cost Management" from the dropdown menu.
3. **Navigate to Credits.**
   - In the left navigation panel, locate and click on "Credits" under the Billing section.
4. **Enter Your Credit Code.**
   - On the Credits page, click the "Redeem credit" button, and you'll find a field to enter your promotional code. Input your credit code exactly as provided, then click "Redeem credit".
5. **Verify Redemption**
   - After successful redemption, you'll see a confirmation message. Your credits will appear in the Credits section with details including the amount, expiration date, and applicable services.

You can also follow this step-by-step guide to redeem your AWS credits. Additionally, explore the AWS Free Tier FAQs for more details.
