from typing import List, Dict, Any
import pandas as pd
from io import BytesIO
from utils.text_utils import normalize_text

def parse_xlsx(content_bytes: bytes, doc_title: str) -> List[Dict[str, Any]]:
    """
    Cada fila de cada hoja visible se convierte en un item de texto:
    "Col1: v1; Col2: v2; ..."
    """
    items: List[Dict[str, Any]] = []
    xls = pd.ExcelFile(BytesIO(content_bytes))
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        # Limite suave: primeras ~10-15 columnas
        df = df.iloc[:, :15]
        df = df.dropna(how="all")
        cols = [str(c) for c in df.columns]
        for r_i, row in df.iterrows():
            pairs = []
            for c_name in cols:
                val = row[c_name]
                if pd.isna(val):
                    continue
                txt = normalize_text(str(val))
                if txt:
                    pairs.append(f"{c_name}: {txt}")
            row_text = "; ".join(pairs)
            row_text = normalize_text(row_text)
            if row_text:
                items.append({
                    "text": row_text,
                    "metadata": {"doc_title": doc_title, "source_type": "excel", "sheet": sheet, "row": int(r_i)}
                })
    return items
