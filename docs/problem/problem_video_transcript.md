# Amazon ML Challenge 2026 — Video Transcript

**Source:** `ml_challenge_2026_video.mp4` (YouTube: `VE-YTgrzmA0`)
**Duration:** ~5 min 56 sec
**Transcript via:** tactiq.io

> Note: auto-generated captions, lightly corrected for obvious speech-to-text errors (marked where meaningful).

---

## Introduction (00:00 – 00:29)

**[00:00:00]** Welcome to the Amazon ML Challenge 2026.

**[00:00:03]** This year's problem is business entity resolution — a fundamental and widely encountered problem in real world data.

**[00:00:10]** You will be given business records that arrive from three independent sources, each noisy and inconsistent. And the goal is to determine which of them describe the same real world business.

**[00:00:21]** In this video, I will walk you through the problem statement, the dataset, the scoring, the submission format, and how entries are judged.

---

## Where the data comes from (00:29 – 01:05)

**[00:00:29]** Let's start with where this data comes from. Consider a business signing up on Amazon Business. At signup, we capture its core details such as the business name and address.

**[00:00:38]** To build a richer picture of that same business, we pull in additional information from other data providers. Each of these sources typically comes from a different data vendor with its own formats and conventions.

**[00:00:50]** The difficulty is that these external sources share no common identifier with ours. So the only fields we can rely on are the business name and address.

**[00:00:59]** For this challenge, we have deliberately limited the data to names and addresses.

---

## The core difficulty (01:05 – 01:48)

**[00:01:05]** Here is the core difficulty. The same real world business is described differently by each source. One vendor may write "Acme Robotics Incorporated." Another abbreviates the address, and a third references a nearby landmark.

**[00:01:18]** Source 1 is our clean, deduplicated reference list. Sources 2 and 3 are the noisy fragments that must be reconciled against it, and there is no shared identifier linking them.

**[00:01:26]** Entity resolution is the task of linking these records together — establishing that differently written records all refer to the same business.

**[00:01:36]** For every entity in Source 1, the goal is to find all of its matching records in Sources 2 and 3. A Source 1 entity may match many records, exactly one, or none at all.

---

## Blocking / candidate generation (01:48 – 02:43)

**[00:01:48]** Comparing every Source 1 record against every Source 2 and Source 3 record would be far too expensive at scale. So we first apply **blocking** — sorting records into buckets using a cheap key built from both the name and the address, so that records likely to match land in the same bucket.

**[00:02:04]** Each color here is one such block. Notice that every record carries its own ID, a noisy name, and an address. And records can group either through a similar name or through a shared address.

**[00:02:17]** We follow the orange block built around "Acme Robotics," and the records in it become the candidate matches for that entity.

**[00:02:24]** Blocking favors recall, so the bucket also pulls in look-alikes: a business with a similar name at a different address, and a different business that happens to share an address. The matching model removes those in the next step.

**[00:02:36]** Blocking narrows an enormous number of possible comparisons down to a manageable set of candidate pairs.

---

## The matching model (02:43 – 03:00)

**[00:02:43]** Finally, a matching model scores each candidate pair and keeps only the true matches, discarding the rest.

**[00:02:50]** That produces the two outputs of this challenge: the **candidate pairs** from your blocking stage, and the **final matching results**, which is the file scored on the leaderboard.

---

## The datasets (03:00 – 03:49)

**[00:03:00]** You are provided with two datasets. The **training set** contains records across all three sources together with the ground truth labels — a file that specifies, for each Source 1 business, exactly which Source 2 and Source 3 records it matches.

**[00:03:14]** The **test set** contains the same three sources but no labels, and this is what you generate predictions for.

**[00:03:21]** A note on the label format. The ground truth is stored as **one row per Source 1 entity**. Its ID maps to a comma-separated list of all of its matching IDs. So a single row carries that entity's complete match set, and the list is empty when the entity matches nothing. This mirrors exactly what you submit.

**[00:03:39]** One practical reminder: all files are **tab-separated**, so read them with an explicit tab separator, otherwise the columns will not parse correctly.

---

## Deliverables (03:49 – 04:40)

**[00:03:49]** There are two deliverables throughout the challenge.

**[00:03:51]** You upload a single file, `matching_results.tsv`, containing one row per Source 1 entity with its predicted matches. This is the **only file scored on the leaderboard**.

**[00:04:03]** At the close of the challenge, every team also submits a single archive. It contains the final matches along with `candidate_pairs.tsv` — the candidate set your blocking stage produced before the model narrowed it down. This file is not scored, but it is used to **audit the quality of your blocking**.

**[00:04:20]** The archive also includes your complete runnable pipeline and a methodology document describing your approach. The packages of the top teams are reviewed in detail before final rankings are confirmed.

**[00:04:30]** One reminder: **run the provided validation script before submitting**, so a simple formatting error does not cost you a submission.

---

## How entries are judged (04:40 – 04:59)

**[00:04:40]** Finally, how entries are judged. Submissions are scored using the **macro F-0.5 metric**, which weights precision twice as heavily as recall.

**[00:04:49]** In practical terms, incorrectly merging two different businesses is penalized roughly twice as much as missing a true match. So **when in doubt, it is safer not to merge**.

---

## Recommendations (04:59 – 05:44)

**[00:04:59]** A few recommendations.

**[00:05:01]** **First, understand singletons.** A singleton is a Source 1 entity that has no matching record in Source 2 or Source 3. If you correctly predict an empty list for it, you earn a full score of 1 on that entity. If you predict any match, you score zero. So identifying the businesses with no match is just as important as finding the ones that do match.

**[00:05:20]** **Second, your blocking strategy sets the ceiling on the recall you can achieve.** So invest in it first, because you cannot match a record you never consider.

**[00:05:29]** **And third, pay attention to region-specific patterns** in both names and addresses.

**[00:05:34]** One firm rule: this is a pure machine learning challenge. So **external databases, APIs, and lookups are strictly prohibited**. Use only the provided data.

---

## Closing (05:44 – 05:56)

**[00:05:44]** That is the challenge. Build something you are proud of, resolve those entities, and enjoy the process. We are excited to see what you create.
