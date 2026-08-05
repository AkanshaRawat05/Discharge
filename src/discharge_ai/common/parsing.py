"""
common/parsing.py
=================

Deterministic, multilingual parsers that turn raw document text (or a JSON
payload) into an `ExtractedCase`.

Why deterministic parsing at all when there is an LLM?  Two reasons:

1.  **Safety.**  Patient ids, doses, dates, totals and payment status are
    copied verbatim by code, never re-typed by a model.  The Clinical Extractor
    Agent then calls the LLM through the MCP `discharge-extraction-prompt` to
    fill anything the parser could not confidently read, and LLM values are only
    accepted for fields the parser left empty.
2.  **Availability.**  The pipeline still runs when the LLM is rate-limited or
    offline — it just records a lower-confidence extraction method.

Label and section-header aliases cover English, Spanish, Hindi, German, Dutch
and French, because the same field arrives as "Discharge Date", "Fecha de Alta",
"Ontslagdatum" or "डिस्चार्ज तिथि" depending on the region.
"""

from __future__ import annotations

import re
from typing import Any

from .doc_loader import LoadedDocument
from .schemas import (
    AbnormalLab,
    Bill,
    BillLineItem,
    DocType,
    ExtractedCase,
    LabTest,
    Medication,
)
from .terminology import (
    detect_language,
    normalize_gender,
    normalize_payment_status,
    parse_bool,
)

# --------------------------------------------------------------------------- #
#  Label aliases  (canonical field  ->  labels seen in the wild)
# --------------------------------------------------------------------------- #
FIELD_LABELS: dict[str, tuple[str, ...]] = {
    "patient_id": (
        "patient id", "patient no", "patient number", "id del paciente",
        "patiëntnummer", "patientnummer", "patienten-nr", "patienten nr",
        "patientennummer", "numéro de patient", "रोगी आईडी", "मरीज आईडी",
    ),
    "patient_name": (
        "patient name", "name", "nombre", "naam", "nom", "रोगी का नाम", "नाम",
        "patient", "paciente",
    ),
    "dob": (
        "date of birth", "dob", "fecha de nacimiento", "geboortedatum",
        "geburtsdatum", "date de naissance", "जन्म तिथि",
    ),
    "sex": ("sex", "sexo", "geslacht", "geschlecht", "sexe", "लिंग"),
    "gender": ("gender", "género", "genero", "geslacht", "लिंग"),
    "age": ("age", "edad", "leeftijd", "alter", "âge", "आयु", "उम्र"),
    "address": ("address", "dirección", "direccion", "adres", "adresse", "पता"),
    "admission_date": (
        "admission date", "date of admission", "fecha de ingreso", "opnamedatum",
        "aufnahmedatum", "date d'admission", "भर्ती तिथि", "प्रवेश तिथि",
    ),
    "discharge_date": (
        "discharge date", "date of discharge", "fecha de alta", "ontslagdatum",
        "entlassungsdatum", "date de sortie", "डिस्चार्ज तिथि", "छुट्टी की तिथि",
    ),
    "ward": ("ward", "sala", "afdeling", "station", "service", "वार्ड"),
    "bed_no": ("bed no", "bed no.", "bed", "cama", "bett", "lit", "बेड", "बिस्तर"),
    "service_line": (
        "service line", "línea de servicio", "linea de servicio", "specialisme",
        "fachabteilung", "spécialité", "सेवा", "विभाग",
    ),
    "attending_physician": (
        "attending physician", "treating physician", "médico tratante",
        "medico tratante", "behandelend arts", "behandelnder arzt",
        "médecin traitant", "ordering physician", "médico solicitante",
        "medico solicitante", "aanvragend arts", "उपचार करने वाले चिकित्सक",
        "चिकित्सक",
    ),
    "consulting_doctors": (
        "consulting doctors", "consulting physicians", "médicos consultores",
        "medicos consultores", "consulterende artsen", "konsiliarärzte",
        "médecins consultants", "परामर्शदाता चिकित्सक",
    ),
    "language": (
        "language of record", "record language", "idioma del registro",
        "taal van dossier", "taal van rapport", "sprache", "langue",
        "report language", "idioma del informe", "रिकॉर्ड की भाषा",
        "रिपोर्ट की भाषा",
    ),
    "discharge_approved": (
        "discharge ok", "discharge approved", "alta aprobada",
        "ontslag goedgekeurd", "entlassung genehmigt", "sortie approuvée",
        "डिस्चार्ज स्वीकृत",
    ),
    "discharge_approved_by": (
        "discharge approved by", "approved by", "aprobado por",
        "goedgekeurd door", "genehmigt von", "स्वीकृत द्वारा",
    ),
    # --- lab-specific --------------------------------------------------------
    "lab_name": (
        "performing lab", "laboratory", "laboratorio", "laboratorium",
        "labor", "laboratoire", "प्रयोगशाला", "performing laboratory",
    ),
    "lab_accreditation": (
        "accreditation", "acreditación", "acreditacion", "accreditatie",
        "akkreditierung", "मान्यता",
    ),
    "specimen_collected": (
        "specimen collected", "muestra recolectada", "afnamedatum",
        "probenentnahme", "नमूना संग्रह",
    ),
    "lab_report_date": (
        "reported", "report date", "reportado", "gerapporteerd", "berichtet",
        "रिपोर्ट किया",
    ),
    # --- bill-specific -------------------------------------------------------
    "bill_id": (
        "bill id", "invoice no", "invoice number", "bill no", "factuurnr",
        "factuurnr.", "factuurnummer", "nº de factura", "no. de factura",
        "rechnungsnr", "बिल आईडी",
    ),
    "billing_date": (
        "issue date", "bill date", "billing date", "fecha de emisión",
        "fecha de emision", "factuurdatum", "rechnungsdatum",
        "जारी करने की तिथि",
    ),
    "due_date": (
        "due date", "fecha de vencimiento", "vervaldatum", "fälligkeitsdatum",
        "देय तिथि",
    ),
    "currency": ("currency", "moneda", "valuta", "währung", "devise", "मुद्रा"),
    "payment_status": (
        "payment status", "estado de pago", "betaalstatus", "zahlungsstatus",
        "statut de paiement", "भुगतान स्थिति",
    ),
    "payment_method": (
        "payment method", "método de pago", "metodo de pago", "betaalwijze",
        "zahlungsart", "भुगतान विधि",
    ),
    "hospital_name": (
        "hospital", "hospital name", "vendor", "vendor name", "ziekenhuis",
        "krankenhaus", "अस्पताल",
    ),
}

#  Reverse index: normalised label -> canonical field
_LABEL_INDEX: dict[str, str] = {}
for _field, _labels in FIELD_LABELS.items():
    for _label in _labels:
        _LABEL_INDEX.setdefault(_label, _field)


# --------------------------------------------------------------------------- #
#  Section-header aliases
# --------------------------------------------------------------------------- #
SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "diagnosis": (
        "discharge diagnosis", "final diagnosis", "diagnóstico de alta",
        "diagnostico de alta", "ontslagdiagnose", "entlassungsdiagnose",
        "diagnostic de sortie", "डिस्चार्ज निदान", "अंतिम निदान", "निदान",
    ),
    "allergies": (
        "allergies", "allergy", "alergias", "allergieën", "allergieen",
        "allergien", "allergies connues", "एलर्जी", "adr", "adr / allergies",
    ),
    "prescriptions": (
        "discharge prescriptions", "prescriptions", "medications",
        "discharge medications", "recetas de alta", "recetas",
        "ontslagrecepten", "recepten", "entlassungsrezepte", "medikamente",
        "ordonnances de sortie", "डिस्चार्ज नुस्खे", "दवाएँ", "दवाइयाँ",
    ),
    "follow_up": (
        "follow-up appointment", "follow up appointment", "follow-up",
        "cita de seguimiento", "seguimiento", "vervolgafspraak", "controle",
        "nachsorgetermin", "rendez-vous de suivi", "अनुवर्ती नियुक्ति",
        "फॉलो-अप",
    ),
    "instructions": (
        "discharge instructions", "instructions", "instrucciones de alta",
        "instrucciones", "ontslaginstructies", "instructies",
        "entlassungsanweisungen", "instructions de sortie",
        "डिस्चार्ज निर्देश", "निर्देश",
    ),
    "lab_results": (
        "laboratory results", "lab results", "results",
        "resultados de laboratorio", "resultados", "laboratoriumresultaten",
        "laborergebnisse", "résultats de laboratoire", "प्रयोगशाला परिणाम",
    ),
    "abnormal_labs": (
        "abnormal lab findings", "abnormal findings", "abnormal results",
        "hallazgos anormales de laboratorio", "hallazgos anormales",
        "abnormale bevindingen", "auffällige befunde",
        "résultats anormaux", "असामान्य प्रयोगशाला निष्कर्ष",
        "असामान्य निष्कर्ष",
    ),
    "comment": (
        "comment", "comments", "note", "notes", "remark", "remarks",
        "comentario", "opmerking", "opmerkingen", "kommentar", "टिप्पणी",
    ),
}

_SECTION_INDEX: dict[str, str] = {}
for _section, _headers in SECTION_ALIASES.items():
    for _header in _headers:
        _SECTION_INDEX.setdefault(_header, _section)


# --------------------------------------------------------------------------- #
#  Low-level helpers
# --------------------------------------------------------------------------- #
_SEPARATOR_RE = re.compile(r"^[=\-_*~#\s]{6,}$")
_LABEL_LINE_RE = re.compile(r"^\s*([^:|]{2,60}?)\s*[::]\s*(.*)$")
_NUMBER_RE = re.compile(r"-?\d{1,3}(?:[, ]\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?")

#  Document footers ("End of Laboratory Report — …", "Fin del Documento …")
#  close the current section; otherwise they get absorbed into it as data.
_FOOTER_RE = re.compile(
    r"^\s*(end of|fin del|fin de|einde van|ende des|fin du|generated|generado|"
    r"gegenereerd|thank you|hartelijk dank|gracias|धन्यवाद|दस्तावेज़ का अंत)\b",
    re.IGNORECASE,
)

#  A lab value is a number that starts its own token — this keeps "HbA1c" from
#  being split into test "HbA" + value "1c 6.9 %".
_LAB_VALUE_RE = re.compile(r"(?:^|\s)(-?\d+(?:[.,]\d+)?)")


def _normalise_label(label: str) -> str:
    text = label.strip().lower().rstrip(".:").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _canonical_field(label: str) -> str | None:
    normalised = _normalise_label(label)
    if normalised in _LABEL_INDEX:
        return _LABEL_INDEX[normalised]
    # tolerate "Bed No." / "Patient ID:" style noise
    stripped = normalised.replace(".", "").strip()
    return _LABEL_INDEX.get(stripped)


def _canonical_section(line: str) -> str | None:
    text = _normalise_label(line)
    if not text or len(text) > 70:
        return None
    if text in _SECTION_INDEX:
        return _SECTION_INDEX[text]
    for header, section in _SECTION_INDEX.items():
        if text.startswith(header) or header in text:
            # Avoid matching prose lines that merely mention the word.
            if len(text) <= len(header) + 18:
                return section
    return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = _NUMBER_RE.search(str(value).replace(",", ""))
    return float(match.group(0)) if match else None


def _to_int(value: Any) -> int | None:
    number = _to_float(value)
    return int(number) if number is not None else None


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().strip("-–—").strip()
    if not text or text.lower() in {"n/a", "na", "none", "null", "-", "--"}:
        return None
    return text


def _split_multi(value: str | None) -> list[str]:
    """Split "Dr. A (Endo); Dr. B (PCP)" into individual entries."""
    if not value:
        return []
    parts = re.split(r"[;\n]|,\s*(?=(?:Dr|Dra|Prof|Mr|Ms)\b)", str(value))
    return [p.strip(" -•\t") for p in parts if p and p.strip(" -•\t")]


def _split_pipe_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.split("|")]


# --------------------------------------------------------------------------- #
#  Text sectioniser
# --------------------------------------------------------------------------- #
def split_sections(text: str) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Split a labelled clinical document into header fields + named sections.

    Returns `(header_fields, sections)` where `header_fields` maps canonical
    field names to raw values found before/outside any section, and `sections`
    maps canonical section names to their raw lines.
    """
    header: dict[str, str] = {}
    sections: dict[str, list[str]] = {}
    current: str | None = None

    for raw_line in (text or "").splitlines():
        line = raw_line.rstrip()
        if not line.strip() or _SEPARATOR_RE.match(line.strip()):
            continue

        if _FOOTER_RE.match(line):
            current = None
            continue

        section = _canonical_section(line)
        if section:
            current = section
            sections.setdefault(current, [])
            continue

        label_match = _LABEL_LINE_RE.match(line)
        if label_match:
            field = _canonical_field(label_match.group(1))
            value = label_match.group(2).strip()
            if field and value and field not in header:
                header[field] = value
                continue

        if current:
            sections[current].append(line.strip())

    return header, sections


# --------------------------------------------------------------------------- #
#  Prescription-table parsing
# --------------------------------------------------------------------------- #
_MED_COLUMN_ORDER = (
    "sl_no", "medicine_name", "strength", "dosage", "frequency",
    "route", "period", "remarks", "total_quantity",
)

_MED_HEADER_TOKENS = (
    "sl.no", "sl no", "nr.", "no.", "medicine name", "medicine",
    "nombre medicamento", "medicamento", "geneesmiddel", "medikament",
    "médicament", "दवा", "strength", "concentración", "sterkte", "stärke",
)


def _looks_like_med_header(cells: list[str]) -> bool:
    joined = " ".join(cells).lower()
    return sum(1 for token in _MED_HEADER_TOKENS if token in joined) >= 2


def parse_prescription_lines(lines: list[str]) -> list[Medication]:
    """Parse the pipe-delimited discharge prescription table."""
    medications: list[Medication] = []

    for line in lines:
        if "|" not in line:
            continue
        cells = _split_pipe_row(line)
        if len(cells) < 3 or _looks_like_med_header(cells):
            continue
        if all(re.fullmatch(r"[-=\s]*", cell) for cell in cells):
            continue

        values: dict[str, Any] = {}
        for index, key in enumerate(_MED_COLUMN_ORDER):
            cell = _clean(cells[index]) if index < len(cells) else None
            values[key] = cell

        values["sl_no"] = _to_int(values.get("sl_no")) or (len(medications) + 1)
        if not values.get("medicine_name"):
            continue
        medications.append(Medication(**values))

    return medications


def parse_medications_from_json(items: Any) -> list[Medication]:
    """Normalise a JSON medication list (keys vary by source system)."""
    medications: list[Medication] = []
    if not isinstance(items, list):
        return medications

    for index, item in enumerate(items, start=1):
        if isinstance(item, str):
            medications.append(Medication(sl_no=index, medicine_name=item))
            continue
        if not isinstance(item, dict):
            continue

        lower = {str(k).lower(): v for k, v in item.items()}
        medications.append(
            Medication(
                sl_no=_to_int(lower.get("sl_no") or lower.get("sl") or index) or index,
                medicine_name=_clean(
                    lower.get("medicine_name") or lower.get("name")
                    or lower.get("medicine") or lower.get("drug")
                    or lower.get("medicamento") or lower.get("geneesmiddel")
                ),
                strength=_clean(
                    lower.get("strength") or lower.get("dose") or lower.get("sterkte")
                    or lower.get("concentración") or lower.get("concentracion")
                ),
                dosage=_clean(lower.get("dosage") or lower.get("dosering")),
                frequency=_clean(
                    lower.get("frequency") or lower.get("freq")
                    or lower.get("frecuencia") or lower.get("frequentie")
                ),
                route=_clean(
                    lower.get("route") or lower.get("via")
                    or lower.get("toedieningsweg")
                ),
                period=_clean(
                    lower.get("period") or lower.get("duration")
                    or lower.get("periodo") or lower.get("duur")
                ),
                remarks=_clean(
                    lower.get("remarks") or lower.get("notes")
                    or lower.get("observaciones") or lower.get("opmerkingen")
                ),
                total_quantity=_clean(
                    lower.get("total_quantity") or lower.get("qty")
                    or lower.get("cantidad") or lower.get("totale_hoeveelheid")
                ),
            )
        )
    return medications


# --------------------------------------------------------------------------- #
#  Lab-table parsing
# --------------------------------------------------------------------------- #
_LAB_HEADER_TOKENS = (
    "test", "prueba", "resultado", "result", "resultaat", "units", "unidades",
    "eenheid", "reference", "rango", "referentiebereik", "flag", "indicador",
    "indicator", "परीक्षण", "परिणाम",
)

_ABNORMAL_FLAGS = {
    "high", "low", "abnormal", "critical", "alto", "bajo", "anormal",
    "hoog", "laag", "abnormaal", "hoch", "niedrig", "उच्च", "निम्न", "असामान्य",
}
_NORMAL_FLAGS = {
    "normal", "normaal", "normale", "wnl", "within normal limits", "ok",
    "सामान्य", "unauffällig",
}


def _looks_like_lab_header(cells: list[str]) -> bool:
    joined = " ".join(cells).lower()
    return sum(1 for token in _LAB_HEADER_TOKENS if token in joined) >= 2


def normalize_lab_flag(flag: str | None) -> str:
    if not flag:
        return ""
    text = str(flag).strip().lower()
    if text in _NORMAL_FLAGS:
        return "NORMAL"
    if text in _ABNORMAL_FLAGS:
        return text.upper()
    for marker in _ABNORMAL_FLAGS:
        if marker in text:
            return text.upper()
    for marker in _NORMAL_FLAGS:
        if marker in text:
            return "NORMAL"
    return str(flag).strip().upper()


def parse_lab_lines(lines: list[str]) -> list[LabTest]:
    tests: list[LabTest] = []
    for line in lines:
        if "|" not in line:
            continue
        cells = _split_pipe_row(line)
        if len(cells) < 2 or _looks_like_lab_header(cells):
            continue
        tests.append(
            LabTest(
                test=_clean(cells[0]),
                value=_clean(cells[1]) if len(cells) > 1 else None,
                unit=_clean(cells[2]) if len(cells) > 2 else None,
                reference_range=_clean(cells[3]) if len(cells) > 3 else None,
                flag=normalize_lab_flag(cells[4]) if len(cells) > 4 else "",
            )
        )
    return [t for t in tests if t.test]


def parse_lab_results_from_json(items: Any) -> list[LabTest]:
    """Handle localised lab-result keys (e.g. Hindi 'परीक्षण'/'परिणाम')."""
    tests: list[LabTest] = []
    if not isinstance(items, list):
        return tests

    test_keys = ("test", "prueba", "resultaat_test", "परीक्षण", "name", "naam")
    value_keys = ("value", "result", "resultado", "resultaat", "परिणाम")
    unit_keys = ("unit", "units", "unidades", "eenheid", "इकाई")
    range_keys = (
        "reference_range", "reference", "rango de referencia", "rango",
        "referentiebereik", "संदर्भ सीमा",
    )
    flag_keys = ("flag", "indicador", "indicator", "संकेतक", "status")

    def pick(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
        lowered = {str(k).strip().lower(): v for k, v in record.items()}
        for key in keys:
            if key in lowered:
                return lowered[key]
        return None

    for item in items:
        if isinstance(item, str):
            tests.append(LabTest(test=item))
            continue
        if not isinstance(item, dict):
            continue
        tests.append(
            LabTest(
                test=_clean(pick(item, test_keys)),
                value=_clean(pick(item, value_keys)),
                unit=_clean(pick(item, unit_keys)),
                reference_range=_clean(pick(item, range_keys)),
                flag=normalize_lab_flag(_clean(pick(item, flag_keys))),
            )
        )
    return [t for t in tests if t.test]


_ABNORMAL_ACTION_RE = re.compile(
    r"^(?P<body>.+?)\s*[-–—]\s*(?:action|acción|accion|actie|aktion|कार्रवाई)\s*[::]?\s*"
    r"(?P<action>.+)$",
    re.IGNORECASE,
)


def parse_abnormal_lines(lines: list[str]) -> list[AbnormalLab]:
    """Parse "HbA1c 6.9 % - Action: continue Metformin, recheck in 3 months"."""
    abnormal: list[AbnormalLab] = []
    for line in lines:
        text = line.strip(" -•\t")
        if not text:
            continue
        match = _ABNORMAL_ACTION_RE.match(text)
        if match:
            body = match.group("body").strip()
            action = match.group("action").strip()
        else:
            body, action = text, ""

        value_match = _LAB_VALUE_RE.search(body)
        cut = value_match.start(1) if value_match else None
        test_name = body[:cut].strip() if cut else body
        value = body[cut:].strip() if cut else None
        abnormal.append(
            AbnormalLab(test=_clean(test_name) or body, value=_clean(value), action=action)
        )
    return abnormal


def parse_abnormal_from_json(items: Any) -> list[AbnormalLab]:
    abnormal: list[AbnormalLab] = []
    if not isinstance(items, list):
        return abnormal
    for item in items:
        if isinstance(item, str):
            abnormal.append(AbnormalLab(test=item))
        elif isinstance(item, dict):
            lower = {str(k).lower(): v for k, v in item.items()}
            abnormal.append(
                AbnormalLab(
                    test=_clean(lower.get("test") or lower.get("परीक्षण") or lower.get("name")),
                    value=_clean(lower.get("value") or lower.get("result") or lower.get("परिणाम")),
                    action=_clean(
                        lower.get("action") or lower.get("action_in_ehr")
                        or lower.get("कार्रवाई") or lower.get("actie")
                    ) or "",
                )
            )
    return abnormal


# --------------------------------------------------------------------------- #
#  Bill parsing
# --------------------------------------------------------------------------- #
_BILL_ROW_RE = re.compile(
    r"^(?P<code>[A-Z][A-Z0-9\-_]{2,25})\s+"
    r"(?P<desc>.+?)\s+"
    r"(?P<qty>\d+(?:\.\d+)?)\s+"
    r"(?P<unit>\d+(?:\.\d+)?)\s+"
    r"(?P<total>\d+(?:\.\d+)?)\s*$"
)

_SUBTOTAL_LABELS = ("subtotal", "sub-total", "subtotaal", "उप-योग", "zwischensumme")
_TAX_LABELS = ("tax", "vat", "gst", "btw", "impuesto", "iva", "कर", "steuer", "tva")
_TOTAL_LABELS = (
    "total due", "total amount", "grand total", "total a pagar", "total",
    "totaal te betalen", "totaal", "gesamtbetrag", "कुल देय", "कुल",
)
_GUARANTEE_MARKERS = (
    "insurance guarantee", "guarantee letter", "garantiebrief",
    "carta de garantía", "carta de garantia", "बीमा गारंटी",
    "guarantee of payment", "pre-authorisation", "pre-authorization",
)


def parse_bill_text(text: str) -> Bill:
    """Parse a fixed-width printed hospital bill in any of the six languages."""
    bill = Bill()
    header, _ = split_sections(text)

    bill.bill_id = _clean(header.get("bill_id"))
    bill.patient_id = _clean(header.get("patient_id"))
    bill.billing_date = _clean(header.get("billing_date"))
    bill.currency = _clean(header.get("currency"))
    bill.payment_method = _clean(header.get("payment_method"))
    bill.payment_status = normalize_payment_status(header.get("payment_status"))

    lines = [line.strip() for line in (text or "").splitlines()]

    # Hospital name: the first substantial non-separator line.
    for line in lines:
        if line and not _SEPARATOR_RE.match(line) and ":" not in line and len(line) > 8:
            bill.hospital_name = line.strip()
            break

    for line in lines:
        if not line:
            continue

        row = _BILL_ROW_RE.match(line)
        if row:
            bill.line_items.append(
                BillLineItem(
                    item_code=row.group("code"),
                    description=row.group("desc").strip(),
                    qty=_to_float(row.group("qty")),
                    unit_price=_to_float(row.group("unit")),
                    total=_to_float(row.group("total")),
                )
            )
            continue

        lowered = line.lower()
        if bill.subtotal is None and any(lbl in lowered for lbl in _SUBTOTAL_LABELS):
            bill.subtotal = _to_float(line.split(":")[-1])
        elif bill.tax is None and any(
            re.search(rf"\b{re.escape(lbl)}\b", lowered) for lbl in _TAX_LABELS
        ):
            bill.tax = _to_float(line.split(":")[-1])
        elif any(lbl in lowered for lbl in _TOTAL_LABELS) and ":" in line:
            amount = _to_float(line.split(":")[-1])
            if amount is not None and (
                bill.total_amount is None or amount >= bill.total_amount
            ):
                bill.total_amount = amount
                currency = re.search(r"\b(USD|EUR|INR|GBP|CHF|AED)\b", line)
                if currency and not bill.currency:
                    bill.currency = currency.group(1)

    if any(marker in (text or "").lower() for marker in _GUARANTEE_MARKERS):
        bill.insurance_guarantee_letter = True

    return bill


def parse_bill_json(payload: dict[str, Any]) -> Bill:
    lower = {str(k).lower(): v for k, v in payload.items()}

    line_items: list[BillLineItem] = []
    #  Prefer the English rendition when the source ships one (`line_items_en`).
    raw_items = (
        lower.get("line_items_en")
        or lower.get("line_items")
        or lower.get("items")
        or []
    )
    for item in raw_items if isinstance(raw_items, list) else []:
        if not isinstance(item, dict):
            continue
        cells = {str(k).lower(): v for k, v in item.items()}
        line_items.append(
            BillLineItem(
                item_code=_clean(cells.get("item_code") or cells.get("code")),
                description=_clean(cells.get("description") or cells.get("desc")),
                qty=_to_float(cells.get("qty") or cells.get("quantity")),
                unit_price=_to_float(cells.get("unit_price") or cells.get("rate")),
                total=_to_float(cells.get("total") or cells.get("amount")),
            )
        )

    guarantee = lower.get("insurance_guarantee_letter")
    if guarantee is None:
        blob = " ".join(
            str(lower.get(key, "")) for key in ("notes", "footer", "payment_method")
        ).lower()
        guarantee = any(marker in blob for marker in _GUARANTEE_MARKERS) or None

    return Bill(
        bill_id=_clean(lower.get("bill_id") or lower.get("invoice_id")),
        patient_id=_clean(lower.get("patient_id")),
        hospital_name=_clean(
            lower.get("vendor_name") or lower.get("hospital_name")
            or lower.get("provider")
        ),
        billing_date=_clean(lower.get("issue_date") or lower.get("billing_date")),
        currency=_clean(lower.get("currency")),
        line_items=line_items,
        subtotal=_to_float(lower.get("subtotal")),
        tax=_to_float(lower.get("tax")),
        total_amount=_to_float(
            lower.get("total_amount") or lower.get("total") or lower.get("total_due")
        ),
        payment_status=normalize_payment_status(lower.get("payment_status")),
        payment_method=_clean(lower.get("payment_method")),
        insurance_guarantee_letter=guarantee,
    )


# --------------------------------------------------------------------------- #
#  Per-document parsers
# --------------------------------------------------------------------------- #
def parse_discharge_document(doc: LoadedDocument) -> dict[str, Any]:
    """Fields extracted from a discharge report (text or JSON)."""
    out: dict[str, Any] = {}

    if doc.data:
        lower = {str(k).lower(): v for k, v in doc.data.items()}
        out.update(
            {
                "patient_id": _clean(lower.get("patient_id")),
                "patient_name": _clean(lower.get("patient_name") or lower.get("name")),
                "dob": _clean(lower.get("dob") or lower.get("date_of_birth")),
                "age": _to_int(lower.get("age")),
                "sex": _clean(lower.get("sex")),
                "gender": normalize_gender(lower.get("gender") or lower.get("sex")),
                "address": _clean(lower.get("address")),
                "admission_date": _clean(lower.get("admission_date")),
                "discharge_date": _clean(lower.get("discharge_date")),
                "ward": _clean(lower.get("ward")),
                "bed_no": _clean(lower.get("bed_no") or lower.get("bed")),
                "service_line": _clean(lower.get("service_line")),
                "attending_physician": _clean(
                    lower.get("attending_physician") or lower.get("treating_physician")
                ),
                "follow_up_appointment": _clean(
                    lower.get("follow_up_appointment")
                    or lower.get("follow_up_appointments")
                ),
                "discharge_instructions": _clean(lower.get("discharge_instructions")),
                "discharge_approved": parse_bool(
                    lower.get("discharge_ok")
                    if lower.get("discharge_ok") is not None
                    else lower.get("discharge_approved")
                ),
                "discharge_approved_by": _clean(
                    lower.get("discharge_approved_by") or lower.get("approved_by")
                ),
                "detected_language": _clean(lower.get("language")),
            }
        )

        consulting = lower.get("consulting_doctors") or lower.get("consultants")
        if isinstance(consulting, list):
            out["consulting_doctors"] = [str(c).strip() for c in consulting if c]
        elif consulting:
            out["consulting_doctors"] = _split_multi(str(consulting))

        diagnosis = lower.get("discharge_diagnosis") or lower.get("diagnosis")
        if isinstance(diagnosis, list):
            out["discharge_diagnosis"] = [str(d).strip() for d in diagnosis if d]
        elif diagnosis:
            out["discharge_diagnosis"] = _split_multi(str(diagnosis))

        allergies = lower.get("allergies") or lower.get("adr_allergy_info")
        if isinstance(allergies, list):
            out["allergies"] = [str(a).strip() for a in allergies if a]
        elif allergies:
            out["allergies"] = _split_multi(str(allergies))

        medications = parse_medications_from_json(
            lower.get("medications") or lower.get("prescriptions")
        )
        if medications:
            out["medications"] = medications

        abnormal = parse_abnormal_from_json(lower.get("abnormal_labs"))
        if abnormal:
            out["abnormal_labs"] = abnormal

        #  Fall through to the text parser for anything the JSON omitted, using
        #  the embedded narrative when present.
        narrative = lower.get("raw_text")
        if isinstance(narrative, str) and narrative.strip():
            out.setdefault("_narrative", narrative)

        return {k: v for k, v in out.items() if v not in (None, [], "")}

    header, sections = split_sections(doc.text)

    out["patient_id"] = _clean(header.get("patient_id"))
    out["patient_name"] = _clean(header.get("patient_name"))
    out["dob"] = _clean(header.get("dob"))
    out["age"] = _to_int(header.get("age"))
    out["sex"] = _clean(header.get("sex"))
    out["gender"] = normalize_gender(header.get("gender") or header.get("sex"))
    out["address"] = _clean(header.get("address"))
    out["admission_date"] = _clean(header.get("admission_date"))
    out["discharge_date"] = _clean(header.get("discharge_date"))
    out["ward"] = _clean(header.get("ward"))
    out["bed_no"] = _clean(header.get("bed_no"))
    out["service_line"] = _clean(header.get("service_line"))
    out["attending_physician"] = _clean(header.get("attending_physician"))
    out["consulting_doctors"] = _split_multi(header.get("consulting_doctors"))
    out["discharge_approved"] = parse_bool(header.get("discharge_approved"))
    out["discharge_approved_by"] = _clean(header.get("discharge_approved_by"))
    out["detected_language"] = _clean(header.get("language"))

    diagnosis_lines = [
        re.sub(r"^\s*\d+[.)]\s*", "", line).strip(" -•")
        for line in sections.get("diagnosis", [])
    ]
    out["discharge_diagnosis"] = [line for line in diagnosis_lines if line]

    out["allergies"] = [
        line.strip(" -•") for line in sections.get("allergies", []) if line.strip(" -•")
    ]
    out["medications"] = parse_prescription_lines(sections.get("prescriptions", []))

    follow_up = " ".join(
        line.strip(" -•") for line in sections.get("follow_up", [])
    ).strip()
    out["follow_up_appointment"] = _clean(follow_up)

    instructions = "\n".join(
        line for line in sections.get("instructions", [])
    ).strip()
    out["discharge_instructions"] = _clean(instructions)

    #  "Discharge OK: YES" with a named attending physician is the treating
    #  physician's sign-off in this dataset's paper forms.
    if out.get("discharge_approved") and not out.get("discharge_approved_by"):
        out["discharge_approved_by"] = out.get("attending_physician")

    return {k: v for k, v in out.items() if v not in (None, [], "")}


def parse_lab_document(doc: LoadedDocument) -> dict[str, Any]:
    out: dict[str, Any] = {}

    if doc.data:
        lower = {str(k).lower(): v for k, v in doc.data.items()}
        out["patient_id"] = _clean(lower.get("patient_id"))
        out["lab_name"] = _clean(
            lower.get("performing_lab") or lower.get("lab_name")
            or lower.get("laboratory")
        )
        out["lab_vendor_name"] = _clean(
            lower.get("vendor_name") or lower.get("performing_lab")
            or lower.get("lab_name")
        )
        out["lab_report_date"] = _clean(
            lower.get("reported") or lower.get("report_date")
            or lower.get("specimen_collected")
        )
        tests = parse_lab_results_from_json(
            lower.get("lab_results") or lower.get("tests") or lower.get("results")
        )
        if tests:
            out["lab_tests"] = tests
        abnormal = parse_abnormal_from_json(lower.get("abnormal_labs"))
        if abnormal:
            out["abnormal_labs"] = abnormal
        if not out.get("detected_language"):
            out["detected_language"] = _clean(lower.get("language"))
        return {k: v for k, v in out.items() if v not in (None, [], "")}

    header, sections = split_sections(doc.text)
    out["patient_id"] = _clean(header.get("patient_id"))
    out["lab_name"] = _clean(header.get("lab_name"))
    out["lab_vendor_name"] = _clean(header.get("lab_name"))
    out["lab_report_date"] = _clean(
        header.get("lab_report_date") or header.get("specimen_collected")
    )
    out["lab_tests"] = parse_lab_lines(sections.get("lab_results", []))
    out["abnormal_labs"] = parse_abnormal_lines(sections.get("abnormal_labs", []))

    #  A lab report with no explicit "ABNORMAL FINDINGS" section can still carry
    #  HIGH/LOW flags in the results table — surface those as abnormal too.
    if not out["abnormal_labs"]:
        out["abnormal_labs"] = [
            AbnormalLab(
                test=test.test,
                value=f"{test.value} {test.unit or ''}".strip(),
                action="",
            )
            for test in out["lab_tests"]
            if test.is_abnormal
        ]

    return {k: v for k, v in out.items() if v not in (None, [], "")}


def parse_bill_document(doc: LoadedDocument) -> dict[str, Any]:
    bill = parse_bill_json(doc.data) if doc.data else parse_bill_text(doc.text)
    out: dict[str, Any] = {"bill": bill}
    if bill.patient_id:
        out["patient_id"] = bill.patient_id
    if doc.data:
        language = _clean({str(k).lower(): v for k, v in doc.data.items()}.get("language"))
        if language:
            out["detected_language"] = language
    return out


# --------------------------------------------------------------------------- #
#  Assemble the ExtractedCase
# --------------------------------------------------------------------------- #
_PARSERS = {
    DocType.DISCHARGE_REPORT.value: parse_discharge_document,
    DocType.LAB_REPORT.value: parse_lab_document,
    DocType.BILL.value: parse_bill_document,
}


def build_extracted_case(
    patient_id: str, documents: dict[str, LoadedDocument]
) -> ExtractedCase:
    """Merge the per-document parse results into one `ExtractedCase`."""
    case = ExtractedCase(patient_id=patient_id.upper())
    narratives: list[str] = []
    declared_language: str | None = None

    for doc_type, doc in documents.items():
        parser = _PARSERS.get(doc_type)
        case.source_files[doc_type] = doc.name
        case.raw_text[doc_type] = doc.text
        if doc_type not in case.doc_types_present:
            case.doc_types_present.append(doc_type)

        if doc.error:
            case.extraction_notes.append(f"{doc_type}: {doc.error}")
        if doc.ocr_used:
            case.extraction_notes.append(
                f"{doc_type}: text recovered via OCR ({doc.name}) — verify values"
            )
        if not parser:
            continue

        parsed = parser(doc)
        narrative = parsed.pop("_narrative", None)
        if narrative:
            narratives.append(str(narrative))

        language = parsed.pop("detected_language", None)
        if language and not declared_language:
            declared_language = str(language)

        for key, value in parsed.items():
            if value in (None, [], ""):
                continue
            current = getattr(case, key, None)
            #  First writer wins for scalars; lists/dicts fill only when empty.
            if current in (None, "", [], {}) or (
                key == "bill" and not getattr(current, "total_amount", None)
            ):
                setattr(case, key, value)

    if narratives:
        case.raw_text["narrative"] = "\n\n".join(narratives)

    combined = "\n".join(case.raw_text.values())
    case.detected_language = detect_language(combined, declared_language)
    case.bill.patient_id = case.bill.patient_id or case.patient_id
    case.patient_id = case.patient_id or patient_id.upper()

    if case.age is None and case.dob and case.discharge_date:
        case.age = _age_from_dates(case.dob, case.discharge_date)

    return case


def _age_from_dates(dob: str, reference: str) -> int | None:
    """Derive age from DOB — used only when the document omits the age field."""
    date_re = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
    dob_match, ref_match = date_re.search(dob), date_re.search(reference)
    if not (dob_match and ref_match):
        return None

    born = tuple(int(g) for g in dob_match.groups())
    ref = tuple(int(g) for g in ref_match.groups())
    age = ref[0] - born[0] - ((ref[1], ref[2]) < (born[1], born[2]))
    return age if 0 <= age < 130 else None
