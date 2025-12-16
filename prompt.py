from langchain_core.prompts import ChatPromptTemplate

def create_ocr_prompt() -> ChatPromptTemplate:
    system_prompt = (
        "Analyze the provided image and extract the financial data into a **Markdown Table**."
        "The table must have exactly 4 columns: | Description | Note | 2023 | 2022 |.\n"
        "Rules:\n"
        "1. **Notes**: Look closely for the small isolated numbers (like 4, 5, 12). Put them in the 'Note' column.\n"
        "2. **Negatives**: Convert '(123)' to '-123'.\n"
        "3. **Empty Cells**: If a Note is missing, leave the cell empty.\n"
        "4. **Exact Layout**: Do not merge rows. Extract every line item."
    )
    
    image_payload = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_data}"}}]
    return ChatPromptTemplate.from_messages([("system", system_prompt), ("user", image_payload)])