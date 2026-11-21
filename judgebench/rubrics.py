"""The identical baseline rubric is shown to humans and the model judge."""
from .schemas import DIMENSIONS

ANCHORS = {
    "correctness": {
        "1": "Core answer is wrong, contradictory, or unsupported by the supplied source.",
        "2": "Major factual or reasoning errors; only a small portion is correct.",
        "3": "Main idea partly correct, with a meaningful error or unsupported claim.",
        "4": "Core answer and reasoning correct; only a minor imprecision.",
        "5": "All substantive claims and reasoning correct and source-faithful."
    },
    "relevance": {
        "1": "Does not address the task.", "2": "Mostly off topic.",
        "3": "Addresses the task but contains substantial distracting material.",
        "4": "Directly relevant with a small digression.",
        "5": "Every substantive part serves the task; brevity alone is not rewarded."
    },
    "completeness": {
        "1": "Essential requested content is absent.", "2": "Most essential content is missing.",
        "3": "Some essential content is missing.", "4": "All essential content, with a minor omission.",
        "5": "Covers all requested content at the requested scope; extra length earns no credit."
    },
    "instruction_following": {
        "1": "Disregards the central instruction or required output format.",
        "2": "Violates multiple explicit constraints or one central constraint.",
        "3": "Follows the main task but violates an important secondary constraint.",
        "4": "Follows all important instructions with a minor formatting deviation.",
        "5": "Follows all explicit instructions, including format, count, and exclusions."
    }
}


def all_rubrics():
    base = {
        "id": "baseline", "version": "1.0", "anchors": ANCHORS,
        "weights": dict.fromkeys(DIMENSIONS, 0.25),
        "guidance": "Grade dimensions independently. A short answer can be complete. Accept valid "
                    "alternatives to the reference. Use the checklist and source, not surface similarity. "
                    "If the prompt has no explicit formatting constraint, assess task compliance. "
                    "Do not prefer a model, writing style, response length, or position."
    }
    strict = {**base, "id": "strict_correctness", "guidance": base["guidance"] +
              " STRICT VARIANT: Any unsupported substantive factual claim or invalid reasoning step "
              "caps correctness at 2. Do not change the other dimension definitions."}
    return {"baseline": base, "strict_correctness": strict}


def weighted_score(scores, weights):
    return sum(scores[d] * weights[d] for d in DIMENSIONS) / sum(weights.values())


def display_rubric(rubric):
    import json
    return json.dumps(rubric, indent=2)
