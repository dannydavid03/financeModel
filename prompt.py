from langchain_core.prompts import ChatPromptTemplate

def create_ocr_prompt() -> ChatPromptTemplate:
    system_prompt = (
        "Analyze the provided image and extract the financial data into a **Markdown Table**."
        "The table must have exactly 4 columns: | Description | Note | 2023 | 2022 |.\n"
        "Rules:\n"
        "1. **Strict Row Alignment**: Read the image strictly line-by-line. **NEVER** attribute numbers to a header line located above them.\n"
        "2. **Headers**: If a line (e.g., 'Assets', 'Non-current assets') has no values on the same horizontal line, leave the value columns empty.\n"
        "3. **Line Items**: Only assign numbers to the text that sits on the **exact same visual horizontal line** (e.g., 'Property and equipment').\n"
        "4. **Notes**: Look for small isolated numbers (like 4, 12) between the text and the amounts.\n"
        "5. **Dashes**: If a cell is '-', keep it as '-'.\n"
        "6. **No Merging**: Do not merge the header 'Non-current assets' with the values of 'Property and equipment'. They are two separate rows."
    )
    
    image_payload = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_data}"}}]
    return ChatPromptTemplate.from_messages([("system", system_prompt), ("user", image_payload)])