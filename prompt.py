from langchain_core.prompts import ChatPromptTemplate

def create_ocr_prompt() -> ChatPromptTemplate:
    """
    Standard Fixed-Column Prompt (Balance Sheet / P&L)
    Expects exactly 4 columns: Description, Note, Year 1, Year 2.
    """
    system_prompt = (
        "Analyze the provided image and extract the financial data into a **Markdown Table**.\n"
        "The table must have exactly 4 columns: | Description | Note | Current Year | Previous Year |.\n"
        "Rules:\n"
        "1. **Strict Row Alignment**: Read the image strictly line-by-line. \n"
        "2. **Headers**: If a line is just a section header, leave value columns empty.\n"
        "3. **Notes**: Look for small isolated numbers (like 4, 12) between the text and the amounts.\n"
        "4. **Dashes**: If a cell is '-', keep it as '-'.\n"
        "5. **Negative Numbers**: Keep parentheses ( ) if present (e.g., '(500)' remains '(500)')."
    )
    
    image_payload = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_data}"}}]
    return ChatPromptTemplate.from_messages([("system", system_prompt), ("user", image_payload)])

def create_equity_prompt() -> ChatPromptTemplate:
    """
    Enhanced Dynamic Prompt (Statement of Changes in Equity)
    Focused on row completeness and header merging.
    """
    system_prompt = (
        "You are an expert financial data analyst. Analyze the provided image, which is a 'Statement of Changes in Equity'.\n"
        "Extract **ALL** rows of data into a Markdown Table. Do not summarize.\n\n"
        "**CRITICAL EXTRACTION RULES:**\n"
        "1. **Capture Every Row**: You must extract the opening balance (e.g., 'Balance as at 1 Jan'), ALL intermediate movements (e.g., 'Total comprehensive income', 'Dividends', 'Transferred'), AND the closing balance. Do not skip any rows.\n"
        "2. **Merge Headers**: The column `(Accumulated losses)/ retained earnings` must be treated as ONE SINGLE COLUMN. Do not split it. Merge stacked headers (e.g., 'Share' over 'capital') into single strings like 'Share capital'.\n"
        "3. **First Column**: The first column must be 'Description' and contain the row labels (e.g., 'Balance as at...').\n"
        "4. **Data Integrity**: Copy numbers exactly. Use '-' for dashes. Keep parentheses '( )' for negatives.\n"
        "5. **Formatting**: Ensure the Markdown table is valid. | Description | Share Capital | ... | Total |"
    )
    image_payload = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_data}"}}]
    return ChatPromptTemplate.from_messages([("system", system_prompt), ("user", image_payload)])