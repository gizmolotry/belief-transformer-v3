"""
Convert 03_human_led_synthesis.tex to a clean Markdown (.md) file
preserving math environments and formatting footnotes properly.
"""

import re
from pathlib import Path

def parse_brackets(text, start_pos):
    """Parse balanced curly brackets starting from start_pos."""
    depth = 0
    pos = start_pos
    while pos < len(text):
        if text[pos] == '{':
            depth += 1
        elif text[pos] == '}':
            depth -= 1
            if depth == 0:
                return text[start_pos+1:pos], pos
        pos += 1
    return "", pos

def tex_to_md(text):
    # 1. Remove LaTeX comments
    text = re.sub(r'(?<!\\)%.*$', '', text, flags=re.M)
    
    # 2. Extract and mask equations and inline math to protect math symbols
    math_blocks = []
    
    # Placeholders for block math
    block_patterns = [
        r'\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}',
        r'\\begin\{align\*?\}(.*?)\\end\{align\*?\}',
        r'\$\$(.*?)\$\$'
    ]
    for pattern in block_patterns:
        def replace_block(match):
            placeholder = f"__BLOCK_MATH_{len(math_blocks)}__"
            math_blocks.append(f"\n$$\n{match.group(1).strip()}\n$$\n")
            return placeholder
        text = re.sub(pattern, replace_block, text, flags=re.S)
        
    # Placeholders for inline math
    # Matches $...$ but not $$
    inline_pattern = r'\$([^\$]+)\$'
    def replace_inline(match):
        placeholder = f"__INLINE_MATH_{len(math_blocks)}__"
        math_blocks.append(f"${match.group(1).strip()}$")
        return placeholder
    text = re.sub(inline_pattern, replace_inline, text)

    # 3. Handle footnotes (\footnote{...})
    # Parse balanced curly brackets recursively for footnotes to prevent stripping
    footnote_idx = 1
    footnotes = []
    while True:
        match = re.search(r'\\footnote', text)
        if not match:
            break
        start_idx = match.start()
        # Find the opening brace of footnote
        brace_idx = text.find('{', start_idx)
        if brace_idx == -1:
            break
        content, end_idx = parse_brackets(text, brace_idx)
        # Replace footnote in text with a markdown footnote marker
        marker = f"[^{footnote_idx}]"
        footnotes.append(f"[^{footnote_idx}]: {content.strip()}")
        text = text[:start_idx] + marker + text[end_idx+1:]
        footnote_idx += 1

    # 4. Convert Section headers
    text = re.sub(r'\\section\*?\{(.*?)\}', r'# \1', text)
    text = re.sub(r'\\subsection\*?\{(.*?)\}', r'## \1', text)
    text = re.sub(r'\\subsubsection\*?\{(.*?)\}', r'### \1', text)
    
    # 5. Convert basic inline formatting
    text = re.sub(r'\\textbf\{(.*?)\}', r'**\1**', text)
    text = re.sub(r'\\emph\{(.*?)\}', r'*\1*', text)
    text = re.sub(r'\\texttt\{(.*?)\}', r'`\1`', text)
    
    # 6. Convert lists
    text = re.sub(r'\\begin\{itemize\}', '', text)
    text = re.sub(r'\\end\{itemize\}', '', text)
    text = re.sub(r'\\begin\{enumerate\}', '', text)
    text = re.sub(r'\\end\{enumerate\}', '', text)
    
    # Handle itemize items with bold framing headers
    text = re.sub(r'\\item\s+\*\*([^*]+)\*\*\s*\\\\', r'- **\1**:', text)
    text = re.sub(r'\\item\s+', r'- ', text)
    
    # 7. Parse and clean tables into Markdown format
    table_pattern = re.compile(r'\\begin\{table\}(.*?)\\end\{table\}', re.S)
    for table_match in table_pattern.finditer(text):
        table_raw = table_match.group(1)
        caption_match = re.search(r'\\caption\{(.*?)\}', table_raw)
        caption = caption_match.group(1) if caption_match else "Table"
        
        # Find tabular block
        tabular_match = re.search(r'\\begin\{tabular\}\{([lcr| ]+)\}(.*?)\\end\{tabular\}', table_raw, re.S)
        if tabular_match:
            alignments = tabular_match.group(1)
            body = tabular_match.group(2)
            
            # Parse rows
            rows = []
            for line in body.strip().split(r'\\'):
                line = line.strip()
                if not line or any(x in line for x in ('\\toprule', '\\midrule', '\\bottomrule', '\\hline')):
                    continue
                # Split cells
                cells = [c.strip() for c in line.split('&')]
                # Remove common LaTeX formatting from cell values
                cells = [re.sub(r'\\[a-zA-Z]+', '', c).strip() for c in cells]
                rows.append(cells)
            
            # Construct MD table
            if rows:
                md_table = f"\n**{caption}**\n\n"
                headers = rows[0]
                md_table += "| " + " | ".join(headers) + " |\n"
                
                # Alignments separator
                separators = []
                for a in alignments:
                    if a == 'l': separators.append(':---')
                    elif a == 'r': separators.append('---:')
                    elif a == 'c': separators.append(':---:')
                while len(separators) < len(headers):
                    separators.append('---')
                md_table += "| " + " | ".join(separators) + " |\n"
                
                for row in rows[1:]:
                    while len(row) < len(headers):
                        row.append("")
                    md_table += "| " + " | ".join(row) + " |\n"
                
                text = text.replace(table_match.group(0), md_table + "\n")
                
    # 8. Clean up specific LaTeX environments and artifacts in non-math text
    text = re.sub(r'\\begin\{description\}', '', text)
    text = re.sub(r'\\end\{description\}', '', text)
    text = re.sub(r'\\item\[(.*?)\]', r'- **\1**:', text)
    
    # Strip citation and reference command structures but leave labels cleanly readable
    text = re.sub(r'\\citep\{(.*?)\}', r'[\1]', text)
    text = re.sub(r'\\citet\{(.*?)\}', r'\1', text)
    text = re.sub(r'\\citeyearpar\{(.*?)\}', r'(\1)', text)
    text = re.sub(r'\\ref\{(.*?)\}', r'[\1]', text)
    text = re.sub(r'\\label\{(.*?)\}', '', text)
    text = re.sub(r'\\url\{(.*?)\}', r'<\1>', text)
    
    # Strip remaining formatting commands (excluding math keywords which are protected)
    text = re.sub(r'\\newpage', '', text)
    text = re.sub(r'\\noindent', '', text)
    text = re.sub(r'\\centering', '', text)
    text = re.sub(r'\\doublespacing', '', text)
    text = re.sub(r'\\maketitle', '', text)
    text = re.sub(r'\\input\{.*?\}', '', text)
    text = re.sub(r'\\newpage', '', text)
    text = re.sub(r'\\vspace\{.*?\}', '', text)
    text = re.sub(r'\\hspace\{.*?\}', '', text)
    text = re.sub(r'\\addtolength\{.*?\}\{.*?\}', '', text)
    text = re.sub(r'\\setlength\{.*?\}\{.*?\}', '', text)
    text = re.sub(r'\\bibliographystyle\{.*?\}', '', text)
    text = re.sub(r'\\bibliography\{.*?\}', '', text)
    
    # Strip any other random formatting keywords that aren't inside math placeholders
    # (Safe because math is masked)
    text = re.sub(r'\\[a-zA-Z]+\*?(?:\[.*?\])?', '', text)
    
    # Restore the math blocks exactly as they were
    for i, math_content in enumerate(math_blocks):
        text = text.replace(f"__BLOCK_MATH_{i}__", math_content)
        text = text.replace(f"__INLINE_MATH_{i}__", math_content)
        
    text = text.replace(r'\\', '\n')
    text = text.replace('``', '"').replace("''", '"')
    
    # Add footnotes at the bottom of the document
    if footnotes:
        text += "\n\n## Footnotes\n\n" + "\n".join(footnotes) + "\n"
        
    # Strip double blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text

def main():
    root_dir = Path(__file__).parent.parent
    tex_file = root_dir / "paper" / "concatenated_drafts" / "03_human_led_synthesis.tex"
    md_file = root_dir / "paper" / "concatenated_drafts" / "03_human_led_synthesis.md"
    
    if not tex_file.exists():
        raise FileNotFoundError(f"Source file not found: {tex_file}")
        
    with open(tex_file, "r", encoding="utf-8") as f:
        tex_content = f.read()
        
    print("Converting LaTeX to Markdown...")
    md_content = tex_to_md(tex_content)
    
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print(f"Successfully wrote token-efficient Markdown: {md_file}")

if __name__ == "__main__":
    main()
