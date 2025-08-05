import json
import re



# -----------------------------------------------------------------------------
# JSON extraction helpers
# -----------------------------------------------------------------------------
def extract_json_content(response_str: str, key: str) -> str:
    # 1) Normalize and strip single quotes / ```json fences
    s = response_str.strip()
    if s.startswith("'") and s.endswith("'"):
        s = s[1:-1].strip()
    if s.startswith("```json"):
        s = s.split("```json",1)[1].rsplit("```",1)[0].strip()

    # 2) Auto-balance braces
    opens, closes = s.count("{"), s.count("}")
    if opens > closes:
        s += "}" * (opens - closes)

    # 3) Ensure the value string is closed *before* any trailing brace(s)
    key_pat = rf'"{re.escape(key)}"\s*:\s*"'
    m_key = re.search(key_pat, s)
    if m_key:
        val_start = m_key.end() - 1  # index of the opening "
        # look for a subsequent unescaped "
        if not re.search(r'(?<!\\)"', s[val_start+1:]):
            # Fallback: split off trailing braces
            core_tail_match = re.match(r'^(.*?)(\}*)\s*$', s, flags=re.DOTALL)
            if core_tail_match:
                core, tail = core_tail_match.groups()
            else:
                core, tail = s, ''
            s = core + '"' + tail

    # 4) Try a real JSON parse
    try:
        return json.loads(s)[key]
    except (json.JSONDecodeError, KeyError):
        # 5) Fallback regex that properly skips over \"…
        pat = rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"'
        m = re.search(pat, s, flags=re.DOTALL)
        if not m:
            raise ValueError(f"Could not extract {key!r}")
        # m.group(1) is the raw JSON string content; wrap in quotes and let json.loads handle escapes
        return json.loads(f'"{m.group(1)}"')