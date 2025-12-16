from pydantic import BaseModel, Field
from typing import List, Optional

class FinancialRow(BaseModel):
    # We ask for individual columns by position, forcing a left-to-right scan
    column_1_description: str = Field(
        ..., 
        description="The text content of the first column (Line Item Name)."
    )
    column_2_note: Optional[str] = Field(
        None, 
        description="The content of the second column. LOOK CAREFULLY. This is usually a single small integer (e.g., '4', '12', '15') positioned between the description and the amounts. If the space is truly empty, return null."
    )
    column_3_2023: Optional[str] = Field(
        ..., 
        description="The content of the third column (2023 Amount). If (value), return -value."
    )
    column_4_2022: Optional[str] = Field(
        ..., 
        description="The content of the fourth column (2022 Amount). If (value), return -value."
    )

class FinancialTable(BaseModel):
    rows: List[FinancialRow] = Field(
        ..., 
        description="List of all rows extracted from the financial statement."
    )