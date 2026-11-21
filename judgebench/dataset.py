"""A deliberately small, hand-authored learning dataset, not a public benchmark."""
from collections import Counter
from pathlib import Path

from .storage import read_json, write_json


def build_dataset(validate=True):
    rows = []

    def add(category, prompt, reference, checklist, provenance="Author-created task; reference derived from the supplied information."):
        index = sum(r["category"] == category for r in rows) + 1
        rows.append(dict(id=f"{category}-{index:02d}", category=category,
                             split="dev" if index <= 4 else "test", prompt=prompt,
                             reference=reference, checklist=checklist,
                             provenance=provenance))

    factual = [
        ("What is the chemical symbol for gold?", "Au", ["Identify Au, not Ag."], "https://www.rsc.org/periodic-table/element/79/gold"),
        ("Which planet is closest to the Sun?", "Mercury", ["Name Mercury."], "https://science.nasa.gov/mercury/"),
        ("What is the SI base unit of electric current?", "The ampere (A).", ["Name ampere; do not substitute volt or watt."], "https://www.bipm.org/en/measurement-units/si-base-units"),
        ("How many bits are in one byte under the modern eight-bit definition?", "8 bits.", ["Answer eight bits."], "Author-curated computing definition; eight-bit byte explicitly stipulated."),
        ("What is the capital of Australia?", "Canberra.", ["Name Canberra, not Sydney or Melbourne."], "https://www.australia.gov.au/"),
        ("In DNA, which base normally pairs with adenine?", "Thymine (T).", ["Answer thymine; uracil is the RNA distractor."], "https://www.genome.gov/genetics-glossary/Base-Pair"),
        ("At one standard atmosphere, what is pure water's boiling point in degrees Celsius?", "Approximately 100 degrees Celsius.", ["Give approximately 100 C under the specified pressure."], "Author-curated elementary physical reference; pressure and purity specified."),
        ("Which process lets plants convert light energy into chemical energy?", "Photosynthesis.", ["Name photosynthesis."], "Author-curated elementary biology reference."),
        ("In a relational database, what does a primary key uniquely identify?", "A row (record) in a table.", ["Identify a row within a table; a key may use multiple columns."], "Author-curated relational database definition."),
        ("What does HTTP status code 404 indicate?", "The requested resource was not found (or its existence is not disclosed).", ["Explain Not Found; do not call it an authentication or server-crash status."], "https://www.rfc-editor.org/rfc/rfc9110.html#name-404-not-found")
    ]
    for prompt, ref, checks, source in factual:
        add("factual_qa", prompt, ref, checks, source)

    reasoning = [
        ("A shirt costs $80. It is discounted by 25%, then a 10% tax is applied to the discounted price. What is the total? Show the calculation.", "$80 * 0.75 = $60; $60 * 1.10 = $66.", ["Apply discount before tax.", "Show calculation and $66 total."]),
        ("All zors are blue. Some blue things are round. Must some zors be round? Explain.", "No. The round blue things need not be zors.", ["Reject the invalid inference.", "Explain why overlap is not guaranteed."]),
        ("A train travels 120 km in 2 hours and 60 km in 1 hour. What is its average speed for the entire trip?", "180 km / 3 h = 60 km/h.", ["Use total distance divided by total time.", "Answer 60 km/h."]),
        ("Three fair coins are tossed independently. What is the probability of exactly two heads? Explain.", "3/8: HHT, HTH, THH out of eight equally likely outcomes.", ["Answer 3/8 or 0.375.", "Give a valid combinatorial or enumeration explanation."]),
        ("A has 5 more books than B. Together they have 29 books. How many does each have? Show your reasoning.", "B + (B+5)=29, so B=12 and A=17.", ["A has 17, B has 12.", "Respect both sum and difference."]),
        ("A bag has 3 red and 2 blue balls. Draw two without replacement. What is the probability both are red?", "(3/5)*(2/4)=3/10.", ["Use without-replacement probabilities.", "Answer 0.3 or 3/10."]),
        ("A task needs 12 worker-hours. Three workers work at equal, constant rates with no overhead. How long does it take?", "12 / 3 = 4 hours.", ["Answer 4 hours under the given assumptions."]),
        ("A clock reads 09:45. What time is it 130 minutes later, using 24-hour format?", "11:55.", ["Add 2 hours and 10 minutes correctly.", "Use 11:55."]),
        ("Every successful job has a log. Job X has a log. Does that prove X succeeded? Explain.", "No. A log is necessary under the premise but not sufficient; unsuccessful jobs may also have logs.", ["Do not affirm the consequent.", "Explain necessary versus sufficient, in any wording."]),
        ("An item rises in price by 20%, then falls by 20%. Relative to the original price, what is the net percentage change? Show an example.", "A 4% decrease: 100 -> 120 -> 96.", ["Apply multiplicative changes.", "State 4% decrease and show a valid example."])
    ]
    for prompt, ref, checks in reasoning:
        add("reasoning", prompt, ref, checks)

    summaries = [
        ("The fictional town of Luma will pilot electric buses on two routes in June. The trial lasts six months. Existing fares will stay the same. A decision about expansion will follow a review of reliability and costs.",
         "Summarize in two sentences.", "Luma will trial electric buses on two routes for six months starting in June, without changing fares. Expansion depends on a review of reliability and costs.", ["Two sentences.", "Include routes, timing, unchanged fares, and conditional expansion."]),
        ("A fictional team tested a cache on 100 requests. Median latency fell from 200 ms to 120 ms. Error rates were unchanged. The team has not measured performance under peak load.",
         "Summarize the result and its main limitation in at most 40 words.", "In a 100-request test, caching reduced median latency from 200 to 120 ms without changing error rates; peak-load performance remains untested.", ["Include latency improvement and unchanged errors.", "Mention untested peak load.", "At most 40 whitespace-separated words."]),
        ("A fictional library extended Saturday hours from 4 pm to 7 pm. Weekday hours did not change. The extension is funded for three months by a local grant, after which attendance will be reviewed.",
         "Give a one-sentence summary.", "The library extended Saturday hours to 7 pm for a three-month grant-funded trial, leaving weekdays unchanged and planning an attendance review.", ["One sentence.", "Preserve temporary funding, Saturday-only change, and review."]),
        ("The fictional Orchard app release adds offline notes. Sync occurs when a connection returns. Photo attachments are not available offline. The release is rolling out gradually over two weeks.",
         "Summarize in exactly two bullet points.", "- Orchard is gradually rolling out offline notes over two weeks, with sync when connectivity returns.\n- Photo attachments are not supported offline.", ["Exactly two bullets.", "Mention rollout, reconnection sync, and photo limitation."]),
        ("A fictional school surveyed 80 students about a later start time. Fifty supported it, twenty opposed it, and ten were undecided. No schedule change has been approved. Staff feedback will be collected next.",
         "Summarize in at most 35 words.", "Of 80 surveyed students, 50 favored a later start, 20 opposed it, and 10 were undecided; no change is approved, and staff feedback is next.", ["Preserve survey counts.", "Do not imply approval.", "Mention staff feedback and obey 35-word cap."]),
        ("A fictional store's revenue grew by 10% last quarter while profit fell by 5%. Management attributed the profit decline to higher shipping costs. It plans to renegotiate delivery contracts but has not projected savings.",
         "Summarize in two sentences without adding predictions.", "Revenue rose 10% last quarter, but profit fell 5%, which management attributed to shipping costs. It plans to renegotiate delivery contracts, with no savings forecast yet.", ["Distinguish revenue and profit.", "Attribute the explanation to management.", "Two sentences, no invented savings."]),
        ("A fictional study assigned 60 volunteers to use a reminder app for four weeks. Participants reported missing fewer tasks. There was no control group, and outcomes were self-reported. Researchers called for a controlled trial.",
         "Summarize the finding and why causal conclusions are limited in two sentences.", "Sixty volunteers reported missing fewer tasks after four weeks using reminders. Without a control group and with self-reported outcomes, the study cannot establish causality, so a controlled trial was recommended.", ["Describe reported improvement and sample/duration.", "Include no control and self-report limits.", "Avoid claiming the app caused improvement."]),
        ("The fictional city of Vale will close the east bridge for maintenance on Sunday from 6 am to noon. Pedestrians may use the west walkway. Buses will follow a posted detour. Emergency access will remain available.",
         "Summarize the practical travel information in at most 45 words.", "Vale's east bridge closes Sunday, 6 am-noon, for maintenance; pedestrians can use the west walkway, buses take a posted detour, and emergency access remains open.", ["Include closure timing, pedestrian route, bus detour, emergency access.", "At most 45 words."]),
        ("The fictional Nimbus service outage began at 10:00 and ended at 10:25. An expired certificate blocked logins. Existing sessions continued to work. Engineers renewed the certificate and added expiry alerts.",
         "Summarize cause, impact, and remediation in exactly three bullets.", "- An expired certificate caused a 25-minute outage from 10:00 to 10:25.\n- New logins were blocked, but existing sessions worked.\n- Engineers renewed the certificate and added expiry alerts.", ["Exactly three bullets.", "Include cause, login-only impact, timing, and both remediations."]),
        ("A fictional recycling pilot collected 12 tonnes of material in May, up from 8 tonnes in April. Contamination increased from 5% to 9%. Organizers will add sorting signs before deciding whether to expand the pilot.",
         "Summarize both the improvement and concern in two sentences.", "Collected recycling rose from 8 tonnes in April to 12 in May, while contamination increased from 5% to 9%. Organizers will add sorting signs before deciding on expansion.", ["Preserve collection and contamination numbers.", "Expansion is undecided; signs are planned.", "Two sentences."])
    ]
    for passage, instruction, ref, checks in summaries:
        add("summarization", f"{instruction}\n\nSource passage:\n{passage}", ref, checks,
            "Original fictional passage authored for JudgeBench; reference checked against passage, not a real-world finding.")

    instructions = [
        ("Return only a JSON object with keys 'city' and 'country' for Paris, France. Use double quotes and no Markdown fences.", '{"city":"Paris","country":"France"}', ["Valid JSON object, exactly two requested keys and values.", "No prose or code fences."]),
        ("Sort these integers ascending and output only a comma-separated list: 9, -2, 5, 0, 5.", "-2,0,5,5,9", ["Correct order with duplicate 5 retained.", "Only comma-separated integers; surrounding whitespace is acceptable."]),
        ("Explain a database index in exactly two sentences for a beginner. Do not use the words 'tree' or 'hash'.", "An index helps a database find rows quickly without checking every row. It uses extra storage and can make writes slower.", ["Exactly two sentences; beginner language.", "Neither forbidden word appears, case-insensitive whole-word match.", "Accurate explanation of an index."]),
        ("Give exactly three bullet points naming renewable energy sources. Each bullet must contain only the source name.", "- Solar\n- Wind\n- Hydropower", ["Exactly three bullets with renewable sources.", "Only source names in bullets; valid alternatives accepted."]),
        ("Translate 'Good morning' into Spanish. Output only the translation.", "Buenos días", ["Correct Spanish translation; terminal punctuation acceptable.", "No explanation or other text."]),
        ("From 'Maya bought 3 apples and Leo bought 5 pears', extract people and quantities. Return only JSON: an array of objects with keys name and quantity. Do not include fruit names.", '[{"name":"Maya","quantity":3},{"name":"Leo","quantity":5}]', ["Valid JSON array with two objects and numeric quantities.", "Correct name/quantity pairs, exactly requested keys, no fruit names or fences."]),
        ("Write a polite one-sentence request to reschedule a meeting. Include the exact phrase 'next Tuesday' and do not apologize.", "Could we please reschedule our meeting to next Tuesday?", ["One polite sentence.", "Includes exact case-sensitive phrase next Tuesday.", "No apology; valid alternatives accepted."]),
        ("Classify each as fruit or vegetable in the everyday culinary sense: apple, carrot, banana. Return exactly three lines in that order, each formatted item: category.", "apple: fruit\ncarrot: vegetable\nbanana: fruit", ["Exactly three lines, original order and specified format.", "Correct culinary categories."]),
        ("Return the first five positive even integers as a JSON array, with no other text.", "[2,4,6,8,10]", ["Valid JSON array [2,4,6,8,10].", "No other text or fences."]),
        ("Summarize 'The deployment failed because its configuration file was missing' in exactly six words. Preserve the cause.", "Missing configuration file caused deployment failure.", ["Exactly six whitespace-separated words.", "Preserve missing configuration file as cause; accept alternative six-word paraphrases."])
    ]
    for prompt, ref, checks in instructions:
        add("instruction_following", prompt, ref, checks)
    return validate_dataset(rows) if validate else rows


def validate_dataset(rows):
    from .schemas import Question
    parsed = [Question.model_validate(q).model_dump() for q in rows]
    if len(parsed) != 40 or len({q["id"] for q in parsed}) != 40:
        raise ValueError("Dataset requires exactly 40 unique questions.")
    counts = Counter(q["category"] for q in parsed)
    if set(counts.values()) != {10} or len(counts) != 4:
        raise ValueError("Each of the four categories requires ten questions.")
    for category in counts:
        if Counter(q["split"] for q in parsed if q["category"] == category) != {"dev": 4, "test": 6}:
            raise ValueError("Require 4 dev / 6 test questions per category.")
    return parsed


def load_dataset(path="data/questions.json"):
    return validate_dataset(read_json(path))


def save_default(path="data/questions.json"):
    if Path(path).exists():
        raise FileExistsError("Dataset exists; edit it intentionally rather than overwriting it.")
    write_json(path, build_dataset())
