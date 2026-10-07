import sys
sys.path.insert(0, r'E:\LegalRAG\data_processing')

from classifier import extract_pdf_lines, LineClassifier
from parser import LegalStructureParser
from pathlib import Path
import types

lines = extract_pdf_lines(Path(r'E:\LegalRAG\data\Văn-bản-hợp-nhất-125-VBHN-VPQH.pdf'))
classifier = LineClassifier()
classified = classifier.classify_all(lines)

parser = LegalStructureParser()
parser.classifier = classifier

# Debug _handle_article
original_handle = parser._handle_article
def debug_handle(line_info, i, classified_list):
    if line_info.is_structural_candidate and line_info.candidate_type == 'article':
        art_num = line_info.features.get('article_number')
        if art_num in [23, 26, 28, 38, 39, 50, 66, 87, 89, 103, 106, 110, 114, 117, 132, 155, 167, 186, 198, 218, 220]:
            print(f'DEBUG _handle_article CALLED: Điều {art_num} at index {i}')
    result = original_handle(line_info, i, classified_list)
    if line_info.is_structural_candidate and line_info.candidate_type == 'article':
        art_num = line_info.features.get('article_number')
        if art_num in [23, 26, 28, 38, 39, 50, 66, 87, 89, 103, 106, 110, 114, 117, 132, 155, 167, 186, 198, 218, 220]:
            print(f'DEBUG _handle_article RETURN: Điều {art_num} -> new_i={result}')
    return result

parser._handle_article = debug_handle

# Debug main loop
def debug_parse(self, lines):
    classified = self.classifier.classify_all(lines)
    self.state = self.state.__class__()
    self.chunks = []
    self.warnings = []

    i = 0
    while i < len(classified):
        line_info = classified[i]
        new_i = i

        if line_info.is_structural_candidate:
            if line_info.candidate_type == "chapter":
                new_i = self._handle_chapter(line_info, i, classified)
            elif line_info.candidate_type == "section":
                new_i = self._handle_section(line_info, i, classified)
            elif line_info.candidate_type == "article":
                if line_info.features.get('article_number') in [23, 26, 28, 38, 39, 50, 66, 87, 89, 103, 106, 110, 114, 117, 132, 155, 167, 186, 198, 218, 220]:
                    print(f'MAIN LOOP: Processing Điều {line_info.features.get("article_number")} at i={i}')
                new_i = self._handle_article(line_info, i, classified)
            elif line_info.candidate_type == "clause":
                new_i = self._handle_clause(line_info, i, classified)
            elif line_info.candidate_type == "point":
                new_i = self._handle_point(line_info, i, classified)

        if new_i == i:
            self._handle_content(line_info)
            new_i = i + 1

        i = new_i

    self._close_all()
    return self.chunks

parser.parse = types.MethodType(debug_parse, parser)

chunks = parser.parse(lines)

articles_found = set()
for c in chunks:
    if c.get('article'):
        m = c['article']
        if m.startswith('Điều '):
            try:
                n = int(m.split()[1])
                articles_found.add(n)
            except:
                pass

print(f"\nArticles found in chunks: {len(articles_found)}")
missing = set(range(1, 221)) - articles_found
print(f"Missing: {sorted(missing)}")