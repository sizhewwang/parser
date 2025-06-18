import os
import re
import csv
import spacy
from fuzzywuzzy import fuzz
from bs4 import BeautifulSoup
from collections import defaultdict

nlp = spacy.load("en_core_web_sm")

SKIP_KEYWORDS = [
    "adjusted", "non-gaap", "core earnings", "pro forma", "reconcile",
    "reconciliation", "excluded from core", "excluding", "discontinued operations"
]

REQUIRED_LABEL_TERMS = [
    "earnings per share", "loss per share", "net income per share",
    "basic earnings", "(loss) earnings per share", "(loss)/earnings per share"
]

CURRENT_PERIOD_KEYWORDS = [
    "quarter ended", "three months ended", "first quarter"
]

def fuzzy_match(text, keywords, threshold=85):
    """
    Check if the input text approximately matches any keyword using fuzzy string matching.

    Args:
        text (str): The input string to match.
        keywords (List[str]): A list of target keywords.
        threshold (int): Fuzzy match score threshold (default: 85).

    Returns:
        bool: True if a fuzzy match exceeds the threshold, else False.
    """
    return any(fuzz.partial_ratio(text, kw) >= threshold for kw in keywords)

def log_eps_selection(filename, method, details, value):
    """
    Append an EPS extraction result to the debug log file for analysis.

    Args:
        filename (str): The name of the input HTML file.
        method (str): The extraction method used ('table', 'regex', or 'ner').
        details (str): Contextual or scoring details for the selection.
        value (float or str): The EPS value selected or 'NA'.
    """
    with open("eps_debug_log.txt", "a") as log_file:
        log_file.write(f"--- {filename} ---\n")
        log_file.write(f"Method: {method}\n")
        log_file.write(f"Details: {details}\n")
        log_file.write(f"Selected EPS: {value}\n\n")

def extract_eps_from_html_tables(soup, filename):
    """
    Extract EPS values from tables in the HTML using structural heuristics and label matching.

    Args:
        soup (BeautifulSoup): Parsed HTML document.
        filename (str): Name of the file being parsed (for logging).

    Returns:
        Tuple[str, float or None, str]: ('table', EPS value or None, context string)
    """
    best_candidate = None
    best_score = -1
    value_scores = defaultdict(lambda: [0, None])

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        headers = []

        for row in rows:
            cells = row.find_all(["td", "th"])
            if not cells or len(cells) < 2:
                continue

            cell_texts = [cell.get_text(strip=True).lower() for cell in cells]
            full_row = " ".join(cell_texts)

            if fuzzy_match(full_row, SKIP_KEYWORDS):
                continue

            if not headers and any("three months ended" in ct for ct in cell_texts):
                headers = cell_texts
                continue

            label = cell_texts[0]
            if not fuzzy_match(label, REQUIRED_LABEL_TERMS):
                continue

            for idx, val in enumerate(cell_texts[1:], start=1):
                clean_val = val.replace(",", "").replace("$", "").replace("(", "-").replace(")", "")
                if re.fullmatch(r"-?\d+\.\d+", clean_val):
                    try:
                        num = float(clean_val)
                        if abs(num) >= 50:
                            continue

                        column_header = headers[idx] if idx < len(headers) else ""
                        context = f"{label} {column_header}".strip()

                        score = 0
                        if 'basic' in label: score += 5
                        if 'diluted' in label: score += 2
                        if 'per share' in label: score += 3
                        if fuzzy_match(context, CURRENT_PERIOD_KEYWORDS, threshold=80): score += 5
                        if fuzzy_match(column_header, CURRENT_PERIOD_KEYWORDS, threshold=80): score += 5
                        if fuzzy_match(label, SKIP_KEYWORDS): score -= 10
                        if 'compared to' in context: score -= 5

                        value_scores[num][0] += score
                        if not value_scores[num][1]:
                            value_scores[num][1] = context

                        if value_scores[num][0] > best_score:
                            best_score = value_scores[num][0]
                            best_candidate = (num, context)
                    except ValueError:
                        continue

    if best_candidate:
        value, context = best_candidate
        log_eps_selection(filename, "robust_table", context, value)
        return ('table', value, context)

    log_eps_selection(filename, "robust_table", "No valid GAAP EPS found", "NA")
    return ('table', None, "")

def extract_eps(text, filename):
    """
    Extract EPS values from raw text using regex-based pattern matching and scoring logic.

    Args:
        text (str): Text content of the HTML document.
        filename (str): Name of the file being parsed (for logging).

    Returns:
        Tuple[str, float or None, str]: ('regex', EPS value or None, context string)
    """
    text = re.sub(r'\((\d+\.\d+)\)', r'-\1', text).replace(",", "").lower()
    raw_candidates = []
    pattern = re.compile(r"(?P<context>.{0,100}?)(?P<value>\(?\$?-?\d+\.\d+\)?)", re.IGNORECASE | re.VERBOSE)

    for match in pattern.finditer(text):
        context = match.group('context').strip()
        raw_value = match.group('value').replace("$", "").replace(" ", "")
        if raw_value.startswith("(") and raw_value.endswith(")"):
            raw_value = "-" + raw_value[1:-1]
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if abs(value) >= 50:
            continue

        if not any(kw in context for kw in ['eps', 'earnings per share', 'loss per share', 'net loss', 'net income', 'per share']):
            continue
        if any(bad_kw in context for bad_kw in ['interest income', 'revenue', 'assets', 'liabilities']):
            continue
        if 'net loss' in context or 'loss of' in context:
            value = -abs(value)

        score = 0
        if 'eps' in context or 'earnings per share' in context: score += 5
        if 'basic' in context: score += 20
        if 'per share' in context: score += 8
        if 'net income' in context: score += 4
        if 'earnings' in context: score += 3
        if 'gaap' in context: score += 2
        if 'loss' in context: score += 2
        if 'diluted' in context: score -= 5
        if fuzzy_match(context, SKIP_KEYWORDS): score -= 10
        if 'compared to' in context or 'versus' in context: score -= 5
        if fuzzy_match(context, CURRENT_PERIOD_KEYWORDS, threshold=80): score += 6

        raw_candidates.append((value, score, context))

    value_scores = defaultdict(lambda: [0, None])
    for val, score, ctx in raw_candidates:
        value_scores[val][0] += score
        if not value_scores[val][1]:
            value_scores[val][1] = ctx

    if value_scores:
        best_val, (agg_score, best_ctx) = max(value_scores.items(), key=lambda kv: kv[1][0])
        log_eps_selection(filename, "regex", f"Top scored: {[(score, val) for val, (score, _) in value_scores.items()]}", best_val)
        return ('regex', best_val, best_ctx)
    else:
        log_eps_selection(filename, "regex", "No matches found", "NA")
        return ('regex', None, "")

def extract_eps_with_ner(text, filename):
    """
    Extract EPS values using Named Entity Recognition to find monetary values in financial context.

    Args:
        text (str): Text content of the HTML document.
        filename (str): Name of the file being parsed (for logging).

    Returns:
        Tuple[str, float or None, str]: ('ner', EPS value or None, context string)
    """
    text = text.replace(",", "")
    doc = nlp(text)
    candidates = []

    for ent in doc.ents:
        if ent.label_ == "MONEY":
            try:
                val = re.sub(r"[\$,()]", "", ent.text).strip()
                num = float(val)
                if abs(num) > 50:
                    continue
                left = text[max(0, ent.start_char - 100):ent.start_char].lower()
                if any(kw in left for kw in ["eps", "per share", "net income", "earnings", "loss", "loss per share"]):
                    if 'adjusted' in left or 'non-gaap' in left or 'discontinued' in left:
                        continue
                    if 'net loss' in left or 'loss of' in left:
                        num = -abs(num)
                    candidates.append((left.strip(), num))
            except:
                continue

    if candidates:
        best = max(candidates, key=lambda tup: len(tup[0]))
        log_eps_selection(filename, "NER", best[0], best[1])
        return ('ner', best[1], best[0])
    else:
        log_eps_selection(filename, "NER", "No candidates", "NA")
        return ('ner', None, "")

def aggregate_votes(candidates):
    """
    Select the final EPS value using a voting system across all extraction methods.

    Args:
        candidates (List[Tuple[str, float or None, str]]): List of method results.

    Returns:
        float or None: Final selected EPS value, or None if no valid value found.
    """
    vote_counter = defaultdict(list)
    method_priority = {"table": 3, "regex": 2, "ner": 1}

    for method, value, _ in candidates:
        if value is None:
            continue
        rounded = round(float(value), 4)
        vote_counter[rounded].append(method)

    if not vote_counter:
        return None

    sorted_votes = sorted(
        vote_counter.items(),
        key=lambda kv: (
            -len(kv[1]),
            -max(method_priority.get(m, 0) for m in kv[1]),
            -kv[0]
        )
    )
    return sorted_votes[0][0]

def parse_file(filepath):
    """
    Process a single HTML file and extract the most likely GAAP EPS value.

    Args:
        filepath (str): Path to the HTML file.

    Returns:
        float or None: Final selected EPS value for the file.
    """
    filename = os.path.basename(filepath)
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
        text = soup.get_text(separator=' ', strip=True)

        candidates = []
        for method_func in [extract_eps_from_html_tables, extract_eps, extract_eps_with_ner]:
            try:
                result = method_func(soup if 'table' in method_func.__name__ else text, filename)
                if result:
                    candidates.append(result)
            except Exception as e:
                continue

        selected = aggregate_votes(candidates)
        log_eps_selection(filename, "voting", f"All candidates: {candidates}", selected)
        return selected

def main(input_folder, output_csv):
    """
    Batch-process all HTML files in the input folder and save EPS results to a CSV file.

    Args:
        input_folder (str): Directory containing .html files.
        output_csv (str): Path to the output CSV file.
    """
    with open("eps_debug_log.txt", "w") as log_file:
        log_file.write("EPS Selection Log\n\n")

    results = []
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith('.html'):
            filepath = os.path.join(input_folder, filename)
            eps = parse_file(filepath)
            results.append((filename, eps if eps is not None else 'NA'))

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'EPS'])
        writer.writerows(results)

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python parser.py <input_folder> <output_csv>")
    else:
        main(sys.argv[1], sys.argv[2])
