import glob
import os
import re
import shutil
import tempfile
import time
import requests
from urllib.parse import quote
from difflib import SequenceMatcher
import xml.etree.ElementTree as ET
import zipfile
import tarfile
import gzip
import json
import traceback
from collections import OrderedDict
import argparse

# Create a global session with a custom User-Agent header.
session = requests.Session()
session.headers.update({
    'User-Agent': 'MyArxivClient/1.0 andre.v.duarte@tecnico.ulisboa.pt'  # Change email as appropriate.
})

def get_with_retries(url, max_retries=3, backoff_factor=1, stream=False):
    """
    Helper function to perform GET requests with retries and exponential backoff.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, stream=stream)
            if response.status_code == 200:
                return response
            else:
                raise Exception(f"HTTP status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            if attempt == max_retries:
                raise Exception(f"Failed to get {url} after {max_retries} attempts. Last error: {e}")
            else:
                sleep_time = backoff_factor * (2 ** (attempt - 1))
                print(f"Attempt {attempt} failed for {url} with error {e}. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
    raise Exception("Unexpected error in get_with_retries")

# ------------- Part 1: arXiv API and Download ----------------

def normalize_text(s):
    """
    Collapse whitespace and replace hyphens with spaces.
    This helps when hyphens cause issues in matching.
    """
    return ' '.join(re.sub(r'[-]', ' ', s).split())

def escape_query_text(s):
    """
    Escape special characters (like colon) that might be misinterpreted
    by the arXiv API query parser. Also, remove hyphens for the API query.
    """
    s = s.replace(":", "\\:")
    s = s.replace("-", " ")
    return s

def get_arxiv_metadata(query, max_results=5, threshold=0.8):
    """
    Retrieve arXiv metadata either by identifier or by title.
    """
    arxiv_id_pattern = re.compile(r'^\d{4}\.\d{4,5}(v\d+)?$')
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    time.sleep(5)
    if arxiv_id_pattern.match(query.strip()):
        url = f"http://export.arxiv.org/api/query?search_query=id:{query.strip()}&start=0&max_results=1"
        response = get_with_retries(url)
        root = ET.fromstring(response.text)
        papers = []
        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns).text.strip()
            summary = entry.find('atom:summary', ns).text.strip()
            authors = [author.find('atom:name', ns).text.strip() for author in entry.findall('atom:author', ns)]
            published = entry.find('atom:published', ns).text.strip()
            arxiv_id = entry.find('atom:id', ns).text.strip().split('/abs/')[-1]
            link_pdf = next((l.attrib['href'] for l in entry.findall('atom:link', ns)
                              if l.attrib.get('type') == 'application/pdf'), None)
            categories = entry.find('atom:category', ns).attrib['term']
            papers.append({
                'title': title,
                'arxiv_id': arxiv_id,
                'arxiv_url': f'https://arxiv.org/abs/{arxiv_id}',
                'pdf_url': link_pdf,
                'authors': authors,
                'summary': summary,
                'published': published,
                'categories': categories,
                'similarity': 1.0
            })
        return papers
    else:
        norm_query = normalize_text(query)
        escaped_query = escape_query_text(query)
        quoted_query = f'"{escaped_query}"'
        encoded_title = quote(quoted_query)
        search_query = f"ti:{encoded_title}"
        url = f"http://export.arxiv.org/api/query?search_query={search_query}&start=0&max_results={max_results}"
        response = get_with_retries(url)
        root = ET.fromstring(response.text)
        papers = []

        def similarity(a, b):
            return SequenceMatcher(None, a, b).ratio()

        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns).text.strip()
            norm_title = normalize_text(title)
            sim_ratio = similarity(norm_query, norm_title)
            if sim_ratio < threshold:
                continue
            summary = entry.find('atom:summary', ns).text.strip()
            authors = [author.find('atom:name', ns).text.strip() for author in entry.findall('atom:author', ns)]
            published = entry.find('atom:published', ns).text.strip()
            arxiv_id = entry.find('atom:id', ns).text.strip().split('/abs/')[-1]
            link_pdf = next((l.attrib['href'] for l in entry.findall('atom:link', ns)
                              if l.attrib.get('type') == 'application/pdf'), None)
            categories = entry.find('atom:category', ns).attrib['term']
            papers.append({
                'title': title,
                'arxiv_id': arxiv_id,
                'arxiv_url': f'https://arxiv.org/abs/{arxiv_id}',
                'pdf_url': link_pdf,
                'authors': authors,
                'summary': summary,
                'published': published,
                'categories': categories,
                'similarity': sim_ratio
            })
        if not papers:
            encoded_title = quote(escape_query_text(query))
            search_query = f"ti:{encoded_title}"
            url = f"http://export.arxiv.org/api/query?search_query={search_query}&start=0&max_results={max_results}"
            response = get_with_retries(url)
            root = ET.fromstring(response.text)
            for entry in root.findall('atom:entry', ns):
                title = entry.find('atom:title', ns).text.strip()
                norm_title = normalize_text(title)
                sim_ratio = similarity(norm_query, norm_title)
                if sim_ratio < threshold:
                    continue
                summary = entry.find('atom:summary', ns).text.strip()
                authors = [author.find('atom:name', ns).text.strip() for author in entry.findall('atom:author', ns)]
                published = entry.find('atom:published', ns).text.strip()
                arxiv_id = entry.find('atom:id', ns).text.strip().split('/abs/')[-1]
                link_pdf = next((l.attrib['href'] for l in entry.findall('atom:link', ns)
                                  if l.attrib.get('type') == 'application/pdf'), None)
                categories = entry.find('atom:category', ns).attrib['term']
                papers.append({
                    'title': title,
                    'arxiv_id': arxiv_id,
                    'arxiv_url': f'https://arxiv.org/abs/{arxiv_id}',
                    'pdf_url': link_pdf,
                    'authors': authors,
                    'summary': summary,
                    'published': published,
                    'categories': categories,
                    'similarity': sim_ratio
                })
        return papers

def determine_extension(file_path):
    """
    Determine the proper extension for the downloaded file.
    Returns one of: ".zip", ".tar.gz", ".tar", or "" if unknown.
    """
    if zipfile.is_zipfile(file_path):
        return ".zip"
    elif tarfile.is_tarfile(file_path):
        with open(file_path, "rb") as f:
            magic = f.read(2)
        if magic == b'\x1f\x8b':  # gzip magic number
            return ".tar.gz"
        else:
            return ".tar"
    else:
        return ""

def download_arxiv_source(arxiv_id, save_path=None):
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    time.sleep(1)
    response = get_with_retries(url, stream=True)
    if response.status_code == 200:
        save_path = save_path or f"{arxiv_id}"
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        ext = determine_extension(save_path)
        if ext and not save_path.endswith(ext):
            new_path = save_path + ext
            os.rename(save_path, new_path)
            save_path = new_path
        print(f"✅ Source file saved to {save_path}")
        return save_path
    else:
        raise Exception(f"❌ Failed to download source for {arxiv_id}. HTTP status: {response.status_code}")

def find_archive_file(arxiv_id, download_dir="."):
    """
    Search for an archive file in download_dir that starts with arxiv_id.
    """
    files = glob.glob(os.path.join(download_dir, f"{arxiv_id}*"))
    if files:
        return files[0]
    return None

# ------------- Part 2: LaTeX Extraction Pipeline ----------------

def strip_comments(text):
    """
    Remove LaTeX comments and comment environments.
    """
    text = re.sub(r'\\begin\{comment\}.*?\\end\{comment\}', '', text, flags=re.DOTALL)
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        if line.lstrip().startswith('%'):
            continue
        line_no_comment = re.split(r'(?<!\\)%', line)[0].rstrip()
        cleaned_lines.append(line_no_comment)
    result = "\n".join(cleaned_lines)
    result = re.sub(r'\n{5,}', '\n\n\n\n', result)
    return result

def find_main_tex_file(archive_path):
    """
    Find the main .tex file in the archive by searching for the one that contains \begin{document}.
    """
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, 'r') as z:
            for file_info in z.infolist():
                if file_info.filename.endswith('.tex') and not file_info.is_dir():
                    with z.open(file_info.filename) as f:
                        try:
                            content = f.read().decode('utf-8', errors='ignore')
                            if '\\begin{document}' in content:
                                return file_info.filename
                        except Exception as e:
                            print(f"Could not read {file_info.filename}: {e}")
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, 'r:*') as t:
            for member in t.getmembers():
                if member.name.endswith('.tex') and member.isfile():
                    f = t.extractfile(member)
                    if f:
                        try:
                            content = f.read().decode('utf-8', errors='ignore')
                            if '\\begin{document}' in content:
                                return member.name
                        except Exception as e:
                            print(f"Could not read {member.name}: {e}")
    elif archive_path.endswith('.gz'):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_name = os.path.basename(archive_path)[:-3]
            output_file = os.path.join(temp_dir, file_name)
            try:
                with gzip.open(archive_path, 'rb') as f_in:
                    with open(output_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                if tarfile.is_tarfile(output_file):
                    with tarfile.open(output_file, 'r:*') as t:
                        for member in t.getmembers():
                            if member.name.endswith('.tex') and member.isfile():
                                f = t.extractfile(member)
                                if f:
                                    try:
                                        content = f.read().decode('utf-8', errors='ignore')
                                        if '\\begin{document}' in content:
                                            return member.name
                                    except Exception as e:
                                        print(f"Could not read {member.name}: {e}")
                else:
                    try:
                        with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            if '\\begin{document}' in content:
                                return file_name
                    except Exception as e:
                        print(f"Could not read {output_file} as text: {e}")
            except Exception as e:
                print(f"Could not process {archive_path}: {e}")
    else:
        raise ValueError("Unsupported archive format. Use a ZIP or tar (possibly compressed) archive, or a .gz file.")
    return None

def extract_tex_files_to_dict(archive_path):
    """
    Extract all .tex files from the archive into a dictionary.
    """
    file_map = {}
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, 'r') as z:
            for file_info in z.infolist():
                if file_info.filename.endswith('.tex') and not file_info.is_dir():
                    key = os.path.normpath(file_info.filename)
                    with z.open(file_info.filename) as f:
                        content = f.read().decode('utf-8', errors='ignore')
                        file_map[key] = content
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, 'r:*') as t:
            for member in t.getmembers():
                if member.name.endswith('.tex') and member.isfile():
                    key = os.path.normpath(member.name)
                    f = t.extractfile(member)
                    if f:
                        content = f.read().decode('utf-8', errors='ignore')
                        file_map[key] = content
    elif archive_path.endswith('.gz'):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_name = os.path.basename(archive_path)[:-3]
            output_file = os.path.join(temp_dir, file_name)
            try:
                with gzip.open(archive_path, 'rb') as f_in:
                    with open(output_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                if tarfile.is_tarfile(output_file):
                    with tarfile.open(output_file, 'r:*') as t:
                        for member in t.getmembers():
                            if member.name.endswith('.tex') and member.isfile():
                                key = os.path.normpath(member.name)
                                f = t.extractfile(member)
                                if f:
                                    content = f.read().decode('utf-8', errors='ignore')
                                    file_map[key] = content
                else:
                    try:
                        with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            file_map[file_name] = content
                    except Exception as e:
                        print(f"Could not read {output_file} as text: {e}")
            except Exception as e:
                print(f"Could not process {archive_path}: {e}")
    else:
        raise ValueError("Unsupported archive format for extracting .tex files.")
    return file_map

def resolve_latex_inputs(file_name, file_map, visited=None, base_path=''):
    """
    Recursively inline \input, \include, and \subfile commands.
    """
    if visited is None:
        visited = set()
    file_name = os.path.normpath(file_name)
    content = file_map.get(file_name)
    if content is None:
        for key in file_map.keys():
            if key.endswith(file_name):
                content = file_map[key]
                file_name = key
                break
    if content is None:
        req_base = os.path.basename(file_name)
        for key in file_map.keys():
            if os.path.basename(key) == req_base:
                content = file_map[key]
                file_name = key
                break
    if content is None:
        print(f"Warning: {file_name} not found.")
        return ''
    visited.add(file_name)

    def replace_commands(match):
        path = match.group(2).strip()
        path_tex = path if path.endswith('.tex') else path + '.tex'
        resolved_path = os.path.normpath(os.path.join(base_path, path_tex))
        if resolved_path in visited:
            return f"% Skipped recursive inclusion of {resolved_path}"
        return resolve_latex_inputs(resolved_path, file_map, visited, os.path.dirname(resolved_path))
    
    pattern = re.compile(r'(?<!%)\\(input|include|subfile)\{([^}]+)\}')
    return pattern.sub(replace_commands, content)

def extract_main_body_only(full_text):
    """
    Extract the main body from \begin{document} until the bibliography or \end{document},
    stripping comments and unnecessary sections.
    """
    doc_match = re.search(r'\\begin\{document\}(.*?)\\end\{document\}', full_text, re.DOTALL)
    if not doc_match:
        return ''
    
    body = doc_match.group(1)
    
    abstract_match = re.search(r'(\\begin\{abstract\}.*?\\end\{abstract\})', body, re.DOTALL)
    if abstract_match:
        body = body[abstract_match.start():]
    
    body = strip_comments(body)
    
    cut_off_patterns = [
        r'\\bibliographystyle\{.*?\}',
        r'\\bibliography\{.*?\}',
        r'\\printbibliography',
        r'\\begin\{thebibliography\}',
        r'\\section\*?\{References\}',
        r'\\section\{References\}',
    ]
    cut_off_regex = re.compile('|'.join(cut_off_patterns), re.IGNORECASE | re.DOTALL)
    cutoff_match = cut_off_regex.search(body)
    
    if cutoff_match:
        return body[:cutoff_match.start()].strip()
    
    return body.strip()

def parse_macro_definitions(full_text):
    """
    Parse macro definitions from the full LaTeX text.
    """
    macros = {}
    pattern_newcommand = re.compile(
        r'\\(?:newcommand|renewcommand)\s*\{\\(\w+)\}\s*(?:\[[^\]]*\])?\s*\{((?:[^{}]|\{[^{}]*\})+)\}'
    )
    pattern_def = re.compile(
        r'\\def\s*\\(\w+)\s*\{((?:[^{}]|\{[^{}]*\})+)\}'
    )
    for match in pattern_newcommand.finditer(full_text):
        macro_name = match.group(1)
        expansion = match.group(2)
        if '#' in expansion:
            continue
        expansion = re.sub(r'\\xspace', '', expansion)
        macros[macro_name] = expansion.strip()
    for match in pattern_def.finditer(full_text):
        macro_name = match.group(1)
        expansion = match.group(2)
        if '#' in expansion:
            continue
        expansion = re.sub(r'\\xspace', '', expansion)
        macros[macro_name] = expansion.strip()
    return macros

def replace_macros(text, macros):
    """
    Replace macro usages in the text with their defined expansions.
    """
    for macro, expansion in macros.items():
        pattern = re.compile(r'\\' + re.escape(macro) + r'\b')
        text = pattern.sub(lambda m: expansion, text)
    return text

def extract_main_paper_body(archive_path):
    """
    Complete pipeline: extract and process the main paper body with macro replacement.
    """
    main_tex = find_main_tex_file(archive_path)
    if not main_tex:
        raise RuntimeError("Main .tex file with \\begin{document} not found.")
    file_map = extract_tex_files_to_dict(archive_path)
    full_latex = resolve_latex_inputs(main_tex, file_map)
    macros = parse_macro_definitions(full_latex)
    main_body = extract_main_body_only(full_latex)
    main_body_replaced = replace_macros(main_body, macros)
    return main_body_replaced

# ------------- Part 3: Integration ----------------

def process_arxiv_paper(paper_identifier, download_dir="."):
    """
    Process a paper either given as an arXiv identifier or as a title.
    Downloads the source, extracts, and returns the main paper body.
    """
    if " " in paper_identifier:
        papers = get_arxiv_metadata(paper_identifier)
        if not papers:
            print(f"No paper found with title: {paper_identifier}")
            return None
        arxiv_id = papers[0]['arxiv_id']
        print(f"Found paper: {papers[0]['title']}\nUsing arXiv id: {arxiv_id}")
    else:
        arxiv_id = paper_identifier

    download_path = os.path.join(download_dir, arxiv_id)
    downloaded_file = download_arxiv_source(arxiv_id, save_path=download_path)
    
    archive_path = find_archive_file(arxiv_id, download_dir)
    if not archive_path:
        raise Exception(f"No archive file found for arXiv id {arxiv_id} in {download_dir}.")
    
    main_body = extract_main_paper_body(archive_path)
    return main_body

def process_papers_from_json(json_file_path, output_json_path=None, recover_id=None):
    """
    Process all papers listed in a JSON file, updating each with its LaTeX source.
    For connection errors ("Connection reset by peer" or "Connection aborted"), the execution stops.
    For other errors, the paper's content is set to null, and the paper ID and error are logged.
    Progress is saved after each paper is processed.
    The error log is written in a finally block so that it's saved even if execution stops.
    """
    if output_json_path is None:
        output_json_path = json_file_path

    with open(json_file_path, 'r') as f:
        data = json.load(f)

    start_index = 0
    if recover_id:
        found = False
        for idx, entry in enumerate(data):
            if entry.get("paper_id") == recover_id:
                start_index = idx
                found = True
                break
        if found:
            print(f"Resuming processing from paper_id {recover_id} at index {start_index}.")
        else:
            print(f"Recover id {recover_id} not found in JSON file. Starting from the beginning.")

    # List to keep track of errors with paper IDs and error messages.
    error_log = []

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            for i in range(start_index, len(data)):
                entry = data[i]
                paper_title = entry.get("paper_title")
                # If no title is provided, set paper_content to null and continue.
                if not paper_title:
                    entry["paper_content"] = None
                    continue

                try:
                    print(f"[{i+1}/{len(data)}] Processing paper: {paper_title}")
                    # Try processing the paper with an inner try/except for recoverable errors.
                    try:
                        main_body = process_arxiv_paper(paper_title, download_dir=temp_dir)
                    except Exception as inner_e:
                        if "Unsupported archive format" in str(inner_e):
                            print(f"⚠️ Unsupported archive format for paper '{paper_title}', setting paper_content to null")
                            main_body = None
                        else:
                            raise inner_e

                    entry["paper_content"] = main_body if main_body else None
                    if main_body:
                        print(f"✅ Successfully processed paper: {paper_title}\n")
                    else:
                        print(f"❌ Paper not found or unsupported archive for: {paper_title}\n")

                except Exception as e:
                    error_str = str(e)
                    # Stop execution for critical connection errors.
                    if "Connection reset by peer" in error_str or "Connection aborted" in error_str:
                        raise Exception(f"Critical connection error encountered for paper '{paper_title}': {error_str}")
                    else:
                        # Log non-critical errors along with the paper_id (or index if missing).
                        paper_id = entry.get("paper_id", f"Index_{i}")
                        error_log.append({"paper_id": paper_id, "error": error_str})
                        print(f"❌ Error processing paper '{paper_title}': {error_str}. Logging and continuing.\n")
                        entry["paper_content"] = None

                # Clean up temporary files in the temp directory.
                for file_path in glob.glob(os.path.join(temp_dir, "*")):
                    if os.path.isfile(file_path):
                        os.remove(file_path)

                # Optionally, save progress after each paper.
                with open(output_json_path, 'w') as f:
                    json.dump(data, f, indent=2)
        finally:
            # This block is executed even if an exception is raised.
            if error_log:
                with open("error_log.json", "w") as elog:
                    json.dump(error_log, elog, indent=2)
                print("Logged paper IDs and errors to 'error_log.json'")

    print(f"✅ Processing complete. Results saved to {output_json_path}")
    return data





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ICLR arXiv papers by year.")
    parser.add_argument("year", type=int, choices=[2024],
                        help="Year of the Conference dataset")
    parser.add_argument("--recover_id", type=str, default=None,
                        help="Paper id from which to resume processing (e.g., '517')")
    args = parser.parse_args()
    
    year = args.year
    json_file_path = f"/media/govprojectstorage/arxiv_test/ICLR_{year}_grouped_reviews.json"

    process_papers_from_json(json_file_path, recover_id=args.recover_id)

