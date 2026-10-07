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

# Debug emit methods
original_emit_article_intro = parser._emit_article_intro
def debug_emit_article_intro():
    print(f'  EMIT article_intro for article={parser.state.article}, content_len={len("".join(parser.state.article_intro))}')
    return original_emit_article_intro()
parser._emit_article_intro = debug_emit_article_intro

original_emit_clause = parser._emit_clause
def debug_emit_clause():
    print(f'  EMIT clause for article={parser.state.article}, clause={parser.state.clause}, content_len={len("".join(parser.state.clause_parts))}')
    return original_emit_clause()
parser._emit_clause = debug_emit_clause

original_emit_clause_intro = parser._emit_clause_intro
def debug_emit_clause_intro():
    print(f'  EMIT clause_intro for article={parser.state.article}, clause={parser.state.clause}')
    return original_emit_clause_intro()
parser._emit_clause_intro = debug_emit_clause_intro

original_emit_point = parser._emit_point
def debug_emit_point():
    print(f'  EMIT point for article={parser.state.article}, clause={parser.state.clause}, point={parser.state.point}')
    return original_emit_point()
parser._emit_point = debug_emit_point

original_close = parser._close_article
def debug_close():
    print(f'CLOSE article: {parser.state.article}')
    return original_close()
parser._close_article = debug_close

# Debug parse
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
                if line_info.features.get('article_number') in [23, 26, 28]:
                    print(f'MAIN: Processing Điều {line_info.features.get("article_number")} at i={i}')
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