import sys
sys.path.insert(0, r'E:\LegalRAG\data_processing')

from classifier import extract_pdf_lines, LineClassifier
from parser import LegalStructureParser
from pathlib import Path

lines = extract_pdf_lines(Path(r'E:\LegalRAG\data\Văn-bản-hợp-nhất-125-VBHN-VPQH.pdf'))
classifier = LineClassifier()
classified = classifier.classify_all(lines)

parser = LegalStructureParser()
parser.classifier = classifier  # Use pre-classified

# Monkey-patch to add debug
original_should = parser._should_start_new_article
def debug_should(line_info, i, classified_list):
    result = original_should(line_info, i, classified_list)
    if line_info.is_structural_candidate and line_info.candidate_type == 'article':
        art_num = line_info.features.get('article_number')
        if art_num in [23, 26, 28, 38, 39, 50, 66, 87, 89, 103, 106, 110, 114, 117, 132, 155, 167, 186, 198, 218, 220]:
            print(f'DEBUG _should: Điều {art_num} -> {result} (prev_inc={line_info.features.get("prev_line_incomplete")}, legal_ref={line_info.features.get("preceded_by_legal_ref")}, has_title={line_info.features.get("has_title")}, next_clause={line_info.features.get("next_line_is_clause")})')
    return result

parser._should_start_new_article = debug_should

# Also patch the parse method to use pre-classified
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

import types
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