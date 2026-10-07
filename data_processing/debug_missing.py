from classifier import extract_pdf_lines, LineClassifier
from pathlib import Path

lines = extract_pdf_lines(Path(r'E:\LegalRAG\data\Văn-bản-hợp-nhất-125-VBHN-VPQH.pdf'))
classifier = LineClassifier()
classified = classifier.classify_all(lines)

missing = [23, 26, 28, 38, 39, 50, 66, 87, 89, 103, 106, 110, 114, 117, 132, 155, 167, 186, 198, 218, 220]

print("=== Articles detected as candidates ===")
for c in classified:
    if c.is_structural_candidate and c.candidate_type == 'article':
        match = c.features.get('article_number')
        if match in missing:
            print(f'FOUND: Điều {match} - title_len={c.features.get("title_length")}, has_title={c.features.get("has_title")}, prev_incomplete={c.features.get("prev_line_incomplete")}, legal_ref={c.features.get("preceded_by_legal_ref")}')

print("\n=== Lines containing missing article numbers ===")
for c in classified:
    if any(f'Điều {n}' in c.text for n in missing):
        print(f'LINE: {c.text[:80]} | candidate={c.is_structural_candidate} type={c.candidate_type}')