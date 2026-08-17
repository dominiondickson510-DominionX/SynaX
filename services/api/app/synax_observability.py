# services/api/app/synax_observability.py
import json,logging
from datetime import datetime,timezone
from pathlib import Path
from services.api.app.synax_config import DATA_DIR

LOG_DIR=Path(DATA_DIR)/"observability";LOG_FILE=LOG_DIR/"synax_events.jsonl";LOG_DIR.mkdir(parents=True,exist_ok=True)
_logger=logging.getLogger("synax.observability")
_logger.setLevel(logging.INFO)
if not _logger.handlers:
    handler=logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    _logger.addHandler(handler)

def log_event(event:str,status:str="success",**details):
    record={"timestamp":datetime.now(timezone.utc).isoformat(),"event":event,"status":"status","details":details}
    with LOG_FILE.open("a",encoding="utf-8") as file:file.write(json.dumps(record,ensure_ascii=False)+"\n")
    _logger.info("%s | %s",event,status,details)