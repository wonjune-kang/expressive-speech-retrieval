import random
import re


# Synonyms / Phrase lists for IEMOCAP, ESD, Expresso.
STYLE_SYNONYMS = {
    "iemocap": {
        "angry": ["angry", "furious", "irritated", "harsh", "irate"],
        "disgust": ["disgusted", "repulsed", "contemptuous", "audibly disdainful", "displeased"],
        "excited": ["excited", "enthusiastic", "eager", "high-energy", "animated"],
        "fear": ["fearful", "scared", "afraid", "trembling", "nervous"],
        "frustrated": ["frustrated", "annoyed", "exasperated", "irritated", "displeased"],
        "happy": ["happy", "joyful", "cheerful", "bright", "delighted"],
        "neutral": ["neutral", "unemotional", "matter-of-fact", "flat", "steady"],
        "sad": ["sad", "gloomy", "melancholy", "somber", "downhearted"],
        "surprise": ["surprised", "shocked", "astonished", "startled", "taken aback"]
    },
    "esd": {
        "Angry": ["angry", "furious", "irritated", "harsh", "irate"],
        "Happy": ["happy", "joyful", "cheerful", "bright", "delighted"],
        "Neutral": ["neutral", "unemotional", "matter-of-fact", "flat", "steady"],
        "Sad": ["sad", "gloomy", "melancholy", "somber", "downhearted"],
        "Surprise": ["surprised", "shocked", "astonished", "startled", "taken aback"]
    },
    "expresso": {
        "angry": ["angry", "furious", "irritated", "harsh", "irate"],
        "awe": ["in awe", "reverent", "amazed", "wonderstruck", "awestruck"],
        "bored": ["bored", "disinterested", "monotone", "lifeless", "unengaged"],
        "calm": ["calm", "soothing", "relaxed", "steady and soft", "gentle"],
        "confused": ["confused", "uncertain", "puzzled", "hesitant", "disoriented"],
        "default": ["neutral", "unemotional", "matter-of-fact", "flat", "steady"],
        "desire": ["longing", "yearning", "desirous", "wanting", "craving"],
        "disgusted": ["disgusted", "repulsed", "contemptuous", "audibly disdainful", "displeased"],
        "enunciated": ["precise", "careful", "clear", "distinct", "articulated"],
        "fast": ["fast-paced", "rushed", "quick", "rapid", "hurried"],
        "fearful": ["fearful", "scared", "afraid", "trembling", "nervous"],
        "happy": ["happy", "joyful", "cheerful", "bright", "delighted"],
        "laughing": ["laughing", "playful", "amused", "giggling", "chuckling"],
        "projected": ["projected", "forceful", "loud and clear", "emphatic", "assertive"],
        "sad": ["sad", "gloomy", "melancholy", "somber", "downhearted"],
        "sarcastic": ["sarcastic", "mocking", "ironic", "dry", "biting"],
        "sleepy": ["sleepy", "drowsy", "yawning", "sluggish", "groggy"],
        "sympathetic": ["sympathetic", "understanding", "warm", "caring", "compassionate"],
        "whisper": ["whispered", "hushed", "quiet", "soft-spoken", "low-volume"]
    },
}

# Prompt templates for IEMOCAP, ESD, Expresso.
PROMPT_TEMPLATES = [
    "The speaker sounds {style}.",
    "Spoken in a {style} tone.",
    "A voice that feels {style}.",
    "Delivered with a {style} quality.",
    "The tone is {style}.",
    "Spoken with a sense of being {style}.",
    "A {style} delivery.",
    "Expressed in a {style} manner.",
    "A vocal tone that is {style}.",
    "Speech that comes across as {style}.",
    "Speech that is {style}."
]


def correct_indefinite_article(template, style_phrase):
    if template.find("a {style}") != -1:
        article = "an" if re.match(r"[aeiouAEIOU]", style_phrase) else "a"
        return template.replace("a {style}", f"{article} {{style}}")
    if template.find("A {style}") != -1:
        article = "An" if re.match(r"[aeiouAEIOU]", style_phrase) else "A"
        return template.replace("A {style}", f"{article} {{style}}")
    return template


def generate_prompt(label, dataset):
    if dataset not in STYLE_SYNONYMS:
        raise ValueError(f"Unknown dataset: {dataset}")

    synonyms = STYLE_SYNONYMS[dataset].get(label)
    if not synonyms:
        raise ValueError(f"Label '{label}' not found in dataset '{dataset}'")

    style_phrase = random.choice(synonyms)
    template = random.choice(PROMPT_TEMPLATES)

    if any(
        style_phrase.startswith(prefix) for prefix in ["spoken", "with", "in"]
    ):
        template = "Speech that is {style}."

    template = correct_indefinite_article(template, style_phrase)

    return template.format(style=style_phrase)
