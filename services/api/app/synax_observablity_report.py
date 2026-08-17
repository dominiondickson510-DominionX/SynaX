# services/api/app/synax_observability_report.py
import json 
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer
from services.api.app.synax_config import DATA_DIR

LOG_DIR=Path(DATA_DIR)/"observability";LOG_FILE=LOG_DIR/"synax_events.jsonl";PDF_FILE=LOG_DIR/"synax_observability_evidence.pdf"

def generate_observability_pdf():
    document=SimpleDocTemplate(str(PDF_FILE),pagesize=A4,rightMargin=15*mm,leftMargin=15*mm,topMargin=15*mm,bottomMargin=15*mm)
    styles=getSampleStyleSheet();title_style=styles["Title"];body_style=styles["BodyText"]
    elements=[Paragraph("SynaX Operational Activity Evidence",title_style),Spacer(1,10)]
    with LOG_FILE.open("r",encoding="utf-8") as file:
        for line in file:
            line=line.strip()
            if not line:continue
            try:record=json.loads(line)
            except json.JSONDecodeError:continue
            timestamp=record.get("timestamp","");event=record.get("event","");status=record.get("status","");details=record.get("details",{})
            text=f"<b>{timestamp}</b><br/>Event: <b>{event}</b><br/>Status: <b>{status}</b><br/>Details: {details}"
            elements.append(Paragraph(text,body_style));elements.append(Spacer(1,8))
    document.build(elements)
    return PDF_FILE