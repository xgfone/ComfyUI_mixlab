# -*- coding: utf-8 -*-


import os.path
import re

_DIR = os.path.dirname(os.path.abspath(__file__))


class PromptLogoCleaner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "brand_file_path": ("STRING", {"default": os.path.join(_DIR, "assets/brand_words.txt")}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("cleaned_prompt",)
    FUNCTION = "clean_prompt"
    CATEGORY = "Prompt/Filter"

    def load_txt_file(self, filepath):
        if not os.path.exists(filepath):
            return []
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return [line.strip() for line in lines if line.strip()]

    def clean_prompt(self, prompt, brand_file_path):
        # Load brand words
        brands = self.load_txt_file(brand_file_path)
        if not brands:
            return (prompt,)

        # Load clothing items
        clothes = self.load_txt_file(os.path.join(_DIR, "assets/clothing_items.txt"))
        if not clothes:
            clothes = ["hoodie", "jacket", "shirt"]

        # Load regex patterns
        raw_patterns = self.load_txt_file(os.path.join(_DIR, "assets/patterns.txt"))
        patterns = []
        for p in raw_patterns:
            pattern = p.replace("{BRANDS}", "|".join(map(re.escape, brands)))
            pattern = pattern.replace("{CLOTHES}", "|".join(map(re.escape, clothes)))
            patterns.append(pattern)

        # Cleaning logic
        sentences = re.split(r"(?<=[.!?]) +", prompt)
        cleaned = []
        for s in sentences:
            for p in patterns:
                # s = re.sub(p, lambda m: m.group(3) if m.lastindex == 3 else "", s, flags=re.IGNORECASE)
                # s = re.sub(p, "", s, flags=re.IGNORECASE)
                # if "(?P<item>" in p:
                #     s = re.sub(p, lambda m: m.group("item"), s, flags=re.IGNORECASE)
                # else:
                #     s = re.sub(p, "", s, flags=re.IGNORECASE)
                if "(?P<item>" in p and "(?P<brand>" in p:

                    def _replace(m):
                        article = m.group("article")
                        item = m.group("item")
                        out = "wearing "
                        if article:
                            out += article + " "
                        out += item
                        return out

                    s = re.sub(p, _replace, s, flags=re.IGNORECASE)
                elif "(?P<item>" in p:
                    s = re.sub(p, lambda m: m.group("item"), s, flags=re.IGNORECASE)
                else:
                    s = re.sub(p, "", s, flags=re.IGNORECASE)

            s = re.sub(r"\s+", " ", s).strip()
            if s:
                cleaned.append(s)

        return (" ".join(cleaned),)
