import re
from nltk import download
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from IPython.display import display, HTML

# Ensure NLTK punkt tokenizer is available
download('punkt', quiet=True)

def find_lcs_positions(X, Y):
    """
    Find indices of the Longest Common Subsequence between two token lists.
    """
    m, n = len(X), len(Y)
    # build DP table
    L = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    # backtrack to get positions
    x_pos, y_pos = [], []
    i, j = m, n
    while i > 0 and j > 0:
        if X[i-1] == Y[j-1]:
            x_pos.append(i-1)
            y_pos.append(j-1)
            i -= 1
            j -= 1
        elif L[i-1][j] >= L[i][j-1]:
            i -= 1
        else:
            j -= 1
    return list(reversed(x_pos)), list(reversed(y_pos))


def build_lcs_to_candidates_map(gold_tokens, candidate_texts):
    """
    Map each token position in gold_tokens to the set of candidate indices matching it.
    """
    token_matches = [set() for _ in gold_tokens]
    for idx, cand in enumerate(candidate_texts):
        cand_tokens = tokenize(cand)
        gold_pos, _ = find_lcs_positions(gold_tokens, cand_tokens)
        for pos in gold_pos:
            token_matches[pos].add(idx)
    return token_matches


def mark_gold_with_multi_overlap(tokens, token_matches, colors, multi_color="#ffcc99"):
    """Highlight gold tokens by candidate overlap."""
    out = []
    for i, tok in enumerate(tokens):
        matches = token_matches[i]
        if matches:
            color = colors[list(matches)[0]] if len(matches) == 1 else multi_color
            out.append(f'<span style="background-color: {color};">{tok}</span>')
        else:
            out.append(tok)
    return " ".join(out)


def mark_candidate(tokens, gold_tokens, color):
    """Highlight candidate tokens present in the LCS with gold."""
    _, cand_pos = find_lcs_positions(gold_tokens, tokens)
    pos_set = set(cand_pos)
    out = []
    for i, tok in enumerate(tokens):
        if i in pos_set:
            out.append(f'<span style="background-color: {color};">{tok}</span>')
        else:
            out.append(tok)
    return " ".join(out)


def tokenize(text):
    """Basic word-and-punctuation tokenization."""
    return re.findall(r"\w+|\S", text)



class TextMetrics:
    """Compute only ROUGE‑L recall between gold and candidate texts."""
    def __init__(self):
        # we ask for only rougeL, with stemming enabled
        self.scorer = rouge_scorer.RougeScorer(
            ['rougeL'],
            use_stemmer=True
        )

    def compute_rouge(self, gold: str, cand: str) -> float:
        # rouge_score returns a dict of Score objects
        scores = self.scorer.score(gold, cand)
        # Score has .precision, .recall, .fmeasure
        return scores['rougeL'].recall


        


def highlight_texts_multicolor(
    gold_text,
    candidate_texts,
    labels=None,
    use_rouge=True
):
    """
    Display gold and candidate texts side-by-side with LCS highlights and ROUGE-L.

    Args:
        gold_text (str): Reference text.
        candidate_texts (List[str]): List of candidate texts.
        labels (List[str], optional): Names for display. Defaults to Candidate 1,2,....
        use_rouge (bool): Whether to compute and display ROUGE-L.
    """
    tokens_gold = tokenize(gold_text)
    token_matches = build_lcs_to_candidates_map(tokens_gold, candidate_texts)

    base_colors = ["#ffff99", "#b3ecff", "#ffd9b3", "#d5b3ff", "#c2f0c2"]
    labels = labels or [f"Candidate {i+1}" for i in range(len(candidate_texts))]

    # Highlight gold text
    highlighted_gold = mark_gold_with_multi_overlap(tokens_gold, token_matches, base_colors)

    # Initialize metrics if needed
    metrics = TextMetrics() if use_rouge else None

    # Build HTML
    html = ['<div style="display: flex; font-family: sans-serif;">']

    # Gold column
    html.append('<div style="flex:1; padding:10px; border-right:1px solid #ccc;">')
    html.append('<h4 style="text-align:center;">Gold</h4>')
    html.append(f'<p style="text-align:justify;">{highlighted_gold}</p>')
    html.append('</div>')

    # Candidate columns
    for idx, cand in enumerate(candidate_texts):
        tok_cand = tokenize(cand)
        highlighted_cand = mark_candidate(tok_cand, tokens_gold, base_colors[idx if idx < len(base_colors) else -1])
        html.append('<div style="flex:1; padding:10px; border-right:1px solid #ccc;">')
        # Title with inline ROUGE-L if available
        if metrics:
            rouge_score = metrics.compute_rouge(gold_text, cand)
            title = f"{labels[idx]} (ROUGE-L = {rouge_score:.4f})"
        else:
            title = labels[idx]
        html.append(f'<h4 style="text-align:center;">{title}</h4>')
        html.append(f'<p style="text-align:justify;">{highlighted_cand}</p>')
        html.append('</div>')

    html.append('</div>')
    display(HTML(''.join(html)))
