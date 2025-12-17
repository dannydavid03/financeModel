import json
import re
from argparse import ArgumentParser
from PIL import Image
import pytesseract

from ocrai import OcrChain

# --- Configuration ---
OLLAMA_HOST = "192.168.3.1"
OLLAMA_PORT = 11434
OLLAMA_MODEL = "gemma3:4b-it-q4_K_M"

def classify_document(image_path):
    """Uses Tesseract to classify document type based on header text."""
    try:
        img = Image.open(image_path)
        w, h = img.size
        header_crop = img.crop((0, 0, w, int(h * 0.25))) # Top 25%
        text = pytesseract.image_to_string(header_crop).lower()
        
        if "equity" in text: return "EQUITY"
        # Added keyword detection for Cash Flows
        if "cash flow" in text or "operating activities" in text: return "CASH"
        if "financial position" in text or "balance sheet" in text: return "BS"
        if "profit" in text or "loss" in text or "income" in text: return "PL"
        return "UNKNOWN"
    except:
        return "UNKNOWN"

def parse_fixed_layout(markdown_text):
    """
    Standard parser for BS/PL and Cash Flows.
    Handles 4 columns (Desc, Note, Cur, Prev) OR 3 columns (Desc, Cur, Prev).
    """
    rows = []
    lines = markdown_text.strip().split('\n')
    for line in lines:
        if '|' in line:
            cells = [c.strip() for c in line.strip('|').split('|')]
            # Filter separators and obvious bad lines
            # Relaxed length check to >= 3 to capture Cash Flow rows
            if len(cells) < 3 or '---' in cells[0] or cells[0].lower() in ['description', 'particulars']:
                continue
            
            # Determine Note presence based on column count and content
            # If 3 columns: cells[0]=Desc, cells[1]=Curr, cells[2]=Prev -> Note is None
            # If 4 columns: cells[0]=Desc, cells[1]=Note, cells[2]=Curr, cells[3]=Prev
            
            note_val = None
            if len(cells) > 3 and len(cells[1]) < 5:
                 note_val = cells[1]
            
            # Robustly grab the last two columns as amounts
            rows.append({
                "line_item": cells[0],
                "note": note_val,
                "amount_current": cells[-2] if len(cells) >= 3 else None, 
                "amount_prev": cells[-1] if len(cells) >= 3 else None
            })
    return rows

def parse_dynamic_layout(markdown_text):
    """
    Robust Dynamic parser for Changes in Equity.
    """
    lines = markdown_text.strip().split('\n')
    headers = []
    rows = []
    header_found = False

    for line in lines:
        if '|' not in line: continue
        
        cells = [c.strip() for c in line.strip('|').split('|')]
        if cells and cells[0] == '': cells.pop(0)
        if cells and cells[-1] == '': cells.pop()

        if not cells: continue
        if any('---' in c for c in cells): continue

        is_candidate_header = any(x in line.lower() for x in ['share', 'capital', 'reserve', 'retained', 'equity', 'total'])
        
        if not header_found:
            if is_candidate_header:
                headers = cells
                if headers and (headers[0] == '' or 'desc' not in headers[0].lower()):
                    headers[0] = "Description"
                header_found = True
            continue

        if header_found:
            row_data = {}
            for i, header in enumerate(headers):
                if i < len(cells):
                    row_data[header] = cells[i]
                else:
                    row_data[header] = None
            
            if row_data.get("Description") and "---" not in row_data["Description"]:
                 rows.append(row_data)
                
    return rows

def main():
    parser = ArgumentParser()
    parser.add_argument("--host", type=str, default=OLLAMA_HOST)
    parser.add_argument("--port", type=int, default=OLLAMA_PORT)
    parser.add_argument("--model", type=str, default=OLLAMA_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--input-image", type=str, required=True)
    parser.add_argument("--output-file", type=str, default="output.json")
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    
    print(f"--- Processing {args.input_image} ---")
    
    # 1. Classify
    doc_type = classify_document(args.input_image)
    print(f"Detected Type: {doc_type}")
    
    # 2. Extract
    prompt_mode = "fixed"
    if doc_type == "EQUITY":
        prompt_mode = "equity"
    elif doc_type == "CASH":
        prompt_mode = "cash"
        
    ocr_chain = OcrChain(model=args.model, base_url=base_url, temperature=args.temperature, mode=prompt_mode)
    
    print("Extracting Markdown...")
    raw_markdown = ocr_chain.invoke(args.input_image)
    print("\n--- Raw LLM Output ---")
    print(raw_markdown)
    print("----------------------\n")

    # 3. Parse
    json_result = {}
    if doc_type == "EQUITY":
        data = parse_dynamic_layout(raw_markdown)
        json_result = {"type": "Statement of Changes in Equity", "rows": data}
    elif doc_type == "CASH":
        # Reuse fixed layout parser; it handles 3 columns (Description, Cur, Prev) automatically
        data = parse_fixed_layout(raw_markdown)
        json_result = {"type": "Statement of Cash Flows", "rows": data}
    else:
        data = parse_fixed_layout(raw_markdown)
        json_result = {"type": "Financial Position / Profit & Loss", "rows": data}

    # 4. Save
    print("Extracted JSON Data:")
    print(json.dumps(json_result, indent=2))
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(json_result, f, indent=2)
    print(f"Saved to {args.output_file}")

if __name__ == "__main__":
    main()