import json
import re
from argparse import ArgumentParser
from ocrai import OcrChain

# --- Configuration ---
OLLAMA_HOST = "192.168.3.1"
OLLAMA_PORT = 11434
OLLAMA_MODEL = "gemma3:4b-it-q4_K_M" 

def parse_markdown_to_json(markdown_text):
    """
    Parses a Markdown table string into a JSON list of dictionaries.
    """
    rows = []
    lines = markdown_text.strip().split('\n')
    
    for line in lines:
        # Check if line looks like a table row (contains pipes)
        if '|' in line:
            # Remove outer pipes and split
            cells = [cell.strip() for cell in line.strip('|').split('|')]
            
            # Skip separator lines (e.g., "---|---|---") and headers
            if len(cells) < 4 or '---' in cells[0] or cells[0].lower() == 'description':
                continue
                
            # Construct Dictionary
            row_data = {
                "line_item": cells[0],
                "note": cells[1] if cells[1] else None, # Handle empty notes
                "amount_2023": cells[2],
                "amount_2022": cells[3]
            }
            rows.append(row_data)
            
    return {"rows": rows}

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
    print(f"Connecting to {base_url}...")

    ocr_chain = OcrChain(model=args.model, base_url=base_url, temperature=args.temperature)
    
    print(f"Processing {args.input_image} (Extracting Markdown)...")
    try:
        # 1. Get Raw Markdown
        raw_markdown = ocr_chain.invoke(args.input_image)
        print("\n--- Raw LLM Output ---")
        print(raw_markdown)
        print("----------------------\n")

        # 2. Convert to JSON
        json_result = parse_markdown_to_json(raw_markdown)

        print("Extracted JSON Data:")
        print(json.dumps(json_result, indent=2))

        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, indent=2)
        print(f"\nSaved to {args.output_file}")

    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()