"""
mock_ehr/data.py
================

Hard-coded EHR records that the Care-Plan & EHR Validation Agent
cross-references against extracted discharge documents.

The deliberate INTENDED MISMATCHES per patient are documented inline so
reviewers can immediately see what each test case is exercising.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Patient demographics
# ---------------------------------------------------------------------------
PATIENTS: dict[str, dict] = {
    "P1001": {
        "patient_id": "P1001", "patient_name": "Alice Johnson",
        "dob": "1968-03-14", "sex": "F",
        "primary_dx": ["E11.9", "I10"],
        "service_line": "General Medicine",
    },
    "P1002": {
        "patient_id": "P1002", "patient_name": "Carlos Mendez",
        "dob": "1955-09-21", "sex": "M",
        "primary_dx": ["I50.9", "I10"],
        "service_line": "General Medicine",
    },
    "P1003": {
        "patient_id": "P1003", "patient_name": "Sarah Williams",
        "dob": "1959-11-02", "sex": "F",
        "primary_dx": ["I21.9", "I25.10", "E11.9"],
        "service_line": "Cardiology",
    },
    "P1004": {
        "patient_id": "P1004", "patient_name": "Raj Kumar",
        "dob": "1981-07-29", "sex": "M",
        "primary_dx": ["J18.9"],
        "service_line": "General Medicine",
    },
    "P1005": {
        "patient_id": "P1005", "patient_name": "Emily Carter",
        "dob": "1995-04-22", "sex": "F",
        "primary_dx": ["J45.909"],
        "service_line": "General Medicine",
    },
    "P1006": {
        "patient_id": "P1006", "patient_name": "John Doe",
        "dob": "1972-01-19", "sex": "M",
        "primary_dx": ["Z12.11", "D50.9"],
        "service_line": "Gastroenterology",
    },
    "P1007": {
        "patient_id": "P1007", "patient_name": "Maria Garcia",
        "dob": "1962-04-11", "sex": "F",
        "primary_dx": ["I50.9"],
        "service_line": "Cardiology",
    },
    "P1008": {
        "patient_id": "P1008", "patient_name": "Li Wei",
        "dob": "1975-12-30", "sex": "M",
        "primary_dx": ["J20.9"],
        "service_line": "General Medicine",
    },
    "P1009": {
        "patient_id": "P1009", "patient_name": "Amir Khan",
        "dob": "1949-02-08", "sex": "M",
        "primary_dx": ["J44.9"],
        "service_line": "Pulmonology",
    },
    "P1010": {
        "patient_id": "P1010", "patient_name": "Olivia Brown",
        "dob": "1990-08-17", "sex": "F",
        "primary_dx": ["J45.909"],
        "service_line": "General Medicine",
    },
    # Handwritten case — clean discharge for community-acquired pneumonia
    "P1011": {
        "patient_id": "P1011", "patient_name": "Sophia Martinez",
        "dob": "1988-09-15", "sex": "F",
        "primary_dx": ["J18.9"],
        "service_line": "General Medicine",
    },
    # Clean UTI case — full prescription table, auto-approve candidate
    "P1012": {
        "patient_id": "P1012", "patient_name": "David Chen",
        "dob": "1992-06-03", "sex": "M",
        "primary_dx": ["N39.0"],
        "service_line": "General Medicine",
    },
    # --- Multilingual batch (P1013–P1017): each trips at least one HITL ---
    # EN / PDF — clean clinically but bill is UNPAID (financial HITL)
    "P1013": {
        "patient_id": "P1013", "patient_name": "Michael Thompson",
        "dob": "1985-02-11", "sex": "M",
        "primary_dx": ["K35.80"],
        "service_line": "General Surgery",
    },
    # ES / handwritten image — medication omission (Loperamide dropped)
    "P1014": {
        "patient_id": "P1014", "patient_name": "Lucía Fernández",
        "dob": "1990-07-19", "sex": "F",
        "primary_dx": ["K52.9"],
        "service_line": "General Medicine",
    },
    # HI / text — low translation confidence guardrail
    "P1015": {
        "patient_id": "P1015", "patient_name": "Ananya Sharma",
        "dob": "1996-11-30", "sex": "F",
        "primary_dx": ["A01.0"],
        "service_line": "General Medicine",
    },
    # DE / JSON — low translation confidence + allergy contradiction
    "P1016": {
        "patient_id": "P1016", "patient_name": "Lukas Müller",
        "dob": "1978-04-05", "sex": "M",
        "primary_dx": ["J18.9"],
        "service_line": "General Medicine",
    },
    # FR / JSON — low translation confidence + medication omission
    "P1017": {
        "patient_id": "P1017", "patient_name": "Camille Laurent",
        "dob": "1983-09-23", "sex": "F",
        "primary_dx": ["N10"],
        "service_line": "General Medicine",
    },
    # EN / text — fully reconciled, perfect record -> auto-approve, summary OK
    "P1018": {
        "patient_id": "P1018", "patient_name": "Robert Anderson",
        "dob": "1970-05-12", "sex": "M",
        "primary_dx": ["E11.9", "I10"],
        "service_line": "General Medicine",
    },
    # EN / text — fully reconciled, paid bill, every field complete ->
    # Low risk, auto-approve, no HITL required.
    "P1019": {
        "patient_id": "P1019", "patient_name": "Thomas Wright",
        "dob": "1972-06-15", "sex": "M",
        "primary_dx": ["E11.9", "I10"],
        "service_line": "General Medicine",
    },
    # ES / PDF — fully reconciled, paid bill; ONLY the address is missing
    # (soft demographic gap, weight 1) -> Low risk, auto-approve, no HITL.
    "P1020": {
        "patient_id": "P1020", "patient_name": "Diego Morales",
        "dob": "1968-02-10", "sex": "M",
        "primary_dx": ["E11.9", "I10"],
        "service_line": "General Medicine",
    },
    # HI / JSON — clinically reconciled (meds, dx, allergy, labs all match) but
    # the bill is UNPAID and BOTH the address and follow-up appointment are
    # missing.  Combined with the Hindi low-translation-confidence guardrail
    # this escalates to HITL (financial + data-entry review).
    "P1021": {
        "patient_id": "P1021", "patient_name": "Rohan Gupta",
        "dob": "1968-03-12", "sex": "M",
        "primary_dx": ["E11.9", "I10"],
        "service_line": "General Medicine",
    },
    # NL / PNG — *** Penicillin allergy on file but the Dutch discharge note
    #     prescribes Amoxicilline -> ALLERGY CONTRADICTION (HARD HITL) ***
    #     Also missing age + doctor details, plus low Dutch translation
    #     confidence -> escalates to High-tier HITL.
    "P1022": {
        "patient_id": "P1022", "patient_name": "Daan Bakker",
        "dob": "1969-08-21", "sex": "M",
        "primary_dx": ["J18.9"],
        "service_line": "General Medicine",
    },
    # EN / handwritten PNG — fully reconciled, paid bill, every mandatory field
    # complete, all labs normal -> Low risk, auto-approve, NO HITL required.
    "P1023": {
        "patient_id": "P1023", "patient_name": "Grace Bennett",
        "dob": "1979-04-18", "sex": "F",
        "primary_dx": ["E11.9", "I10"],
        "service_line": "General Medicine",
    },
    # NL / TXT — *** Penicillin allergy on file but the Dutch discharge note
    #     prescribes Amoxicilline -> ALLERGY CONTRADICTION (HARD HITL) ***
    #     Also MISSING age + doctor details (attending + consulting), plus the
    #     low Dutch translation confidence -> escalates to High-tier HITL.
    "P1024": {
        "patient_id": "P1024", "patient_name": "Bram de Vries",
        "dob": "1971-03-05", "sex": "M",
        "primary_dx": ["J18.9"],
        "service_line": "General Medicine",
    },
}

# ---------------------------------------------------------------------------
# Allergy registry  (the immutable source of truth)
# ---------------------------------------------------------------------------
ALLERGIES: dict[str, list[str]] = {
    "P1001": ["Sulfa"],
    "P1002": ["Penicillin"],
    "P1003": [],
    # *** Penicillin anaphylaxis on file but discharge note prescribes
    #     Amoxicillin-Clavulanate -> ALLERGY CONTRADICTION (HARD HITL) ***
    "P1004": ["Penicillin", "Latex"],
    "P1005": [],
    "P1006": [],
    "P1007": [],
    "P1008": [],
    "P1009": [],
    "P1010": [],
    "P1011": [],
    "P1012": [],
    "P1013": [],
    "P1014": [],
    "P1015": [],
    # *** Penicillin on file but German discharge note prescribes
    #     Amoxicillin -> ALLERGY CONTRADICTION (HARD HITL) ***
    "P1016": ["Penicillin"],
    "P1017": [],
    "P1018": [],
    "P1019": [],
    "P1020": [],
    "P1021": ["Penicillin"],
    # *** Penicillin allergy contradicts the Amoxicilline on the Dutch
    #     discharge note -> ALLERGY CONTRADICTION (HARD HITL) ***
    "P1022": ["Penicillin"],
    # No known drug allergies -> no allergy collision with any discharge med.
    "P1023": [],
    # *** Penicillin allergy contradicts the Amoxicilline on the Dutch
    #     discharge note -> ALLERGY CONTRADICTION (HARD HITL) ***
    "P1024": ["Penicillin"],
}

# ---------------------------------------------------------------------------
# Inpatient medication orders that must reconcile against discharge meds
# ---------------------------------------------------------------------------
MED_ORDERS: dict[str, list[dict]] = {
    "P1001": [
        {"name": "Metformin",    "dose": "500 mg", "frequency": "BID"},
        {"name": "Lisinopril",   "dose": "10 mg",  "frequency": "QD"},
        {"name": "Atorvastatin", "dose": "20 mg",  "frequency": "QHS"},
        {"name": "Aspirin",      "dose": "81 mg",  "frequency": "QD"},
    ],
    "P1002": [
        {"name": "Furosemide",   "dose": "40 mg",  "frequency": "QD"},
        {"name": "Lisinopril",   "dose": "5 mg",   "frequency": "QD"},
        {"name": "Metoprolol",   "dose": "25 mg",  "frequency": "QD"},
        {"name": "Aspirin",      "dose": "81 mg",  "frequency": "QD"},
    ],
    # *** EHR has NO Warfarin order, but discharge note adds Warfarin ->
    #     medication_added + high_risk_med_missing_in_ehr (HARD HITL) ***
    "P1003": [
        {"name": "Aspirin",      "dose": "81 mg",  "frequency": "QD"},
        {"name": "Clopidogrel",  "dose": "75 mg",  "frequency": "QD"},
        {"name": "Atorvastatin", "dose": "80 mg",  "frequency": "QHS"},
        {"name": "Metoprolol",   "dose": "50 mg",  "frequency": "QD"},
        {"name": "Metformin",    "dose": "500 mg", "frequency": "BID"},
    ],
    # Inpatient was Levofloxacin; switched to Amox-Clav (allergy collision)
    "P1004": [
        {"name": "Levofloxacin",   "dose": "750 mg",      "frequency": "QD"},
        {"name": "Acetaminophen",  "dose": "500 mg",      "frequency": "q6h PRN"},
        {"name": "Guaifenesin",    "dose": "400 mg",      "frequency": "q4h PRN"},
    ],
    "P1005": [
        {"name": "Albuterol HFA", "dose": "2 puffs", "frequency": "q4h PRN"},
        {"name": "Fluticasone HFA", "dose": "110 mcg 2 puffs", "frequency": "BID"},
        {"name": "Prednisone",   "dose": "40 mg",  "frequency": "QD x 5 days"},
    ],
    "P1006": [
        {"name": "Pantoprazole",  "dose": "40 mg",  "frequency": "QD"},
        {"name": "Ferrous sulfate", "dose": "325 mg", "frequency": "QD"},
    ],
    # *** EHR has Lisinopril ordered but discharge note OMITS Lisinopril ->
    #     medication_omission ***
    "P1007": [
        {"name": "Furosemide",   "dose": "40 mg",  "frequency": "QD"},
        {"name": "Carvedilol",   "dose": "6.25 mg","frequency": "BID"},
        {"name": "Lisinopril",   "dose": "5 mg",   "frequency": "QD"},
        {"name": "Aspirin",      "dose": "81 mg",  "frequency": "QD"},
    ],
    "P1008": [
        {"name": "Azithromycin",  "dose": "500 mg", "frequency": "QD x 3 days"},
        {"name": "Albuterol HFA", "dose": "2 puffs", "frequency": "q4h PRN"},
    ],
    "P1009": [
        {"name": "Tiotropium",    "dose": "18 mcg",  "frequency": "QD"},
        {"name": "Salbutamol",    "dose": "2 puffs", "frequency": "q4h PRN"},
        {"name": "Prednisone",    "dose": "40 mg",   "frequency": "QD x 5 days"},
    ],
    "P1010": [
        {"name": "Albuterol HFA",  "dose": "2 puffs", "frequency": "q4h PRN"},
        {"name": "Fluticasone HFA","dose": "110 mcg 2 puffs", "frequency": "BID"},
        {"name": "Prednisone",     "dose": "40 mg",   "frequency": "QD x 5 days"},
    ],
    "P1011": [
        {"name": "Amoxicillin",    "dose": "875 mg",  "frequency": "BID x 7 days"},
        {"name": "Azithromycin",   "dose": "500 mg",  "frequency": "QD x 5 days"},
        {"name": "Acetaminophen",  "dose": "500 mg",  "frequency": "q6h PRN"},
    ],
    "P1012": [
        {"name": "Nitrofurantoin", "dose": "100 mg", "frequency": "BID x 5 days"},
        {"name": "Acetaminophen",  "dose": "500 mg", "frequency": "q6h PRN"},
    ],
    # EN post-appendectomy — reconciles fully (only the bill is unpaid)
    "P1013": [
        {"name": "Cefazolin",     "dose": "1 g",    "frequency": "q8h x 2 days"},
        {"name": "Acetaminophen", "dose": "500 mg", "frequency": "q6h PRN"},
        {"name": "Ondansetron",   "dose": "4 mg",   "frequency": "q8h PRN"},
    ],
    # *** ES: EHR ordered Loperamide + Hyoscine but the discharge note
    #     OMITS both -> medication_omission x2 (escalates to Medium HITL) ***
    "P1014": [
        {"name": "Ondansetron",   "dose": "4 mg",   "frequency": "q8h PRN"},
        {"name": "Loperamide",    "dose": "2 mg",   "frequency": "after each loose stool"},
        {"name": "Hyoscine",      "dose": "10 mg",  "frequency": "TID PRN cramps"},
        {"name": "Acetaminophen", "dose": "500 mg", "frequency": "q6h PRN"},
    ],
    # HI typhoid — reconciles fully (only low translation confidence)
    "P1015": [
        {"name": "Azithromycin",  "dose": "500 mg", "frequency": "QD x 7 days"},
        {"name": "Paracetamol",   "dose": "500 mg", "frequency": "q6h PRN"},
    ],
    # DE pneumonia — meds reconcile; allergy contradiction is the trigger
    "P1016": [
        {"name": "Amoxicillin",   "dose": "500 mg", "frequency": "TID x 7 days"},
        {"name": "Paracetamol",   "dose": "500 mg", "frequency": "q6h PRN"},
    ],
    # *** FR pyelonephritis: EHR ordered Nitrofurantoin but discharge note
    #     OMITS it -> medication_omission ***
    "P1017": [
        {"name": "Ciprofloxacin", "dose": "500 mg", "frequency": "BID x 7 days"},
        {"name": "Acetaminophen", "dose": "500 mg", "frequency": "q6h PRN"},
        {"name": "Nitrofurantoin","dose": "100 mg", "frequency": "BID x 5 days"},
    ],
    # Perfect reconciliation — discharge meds match these exactly
    "P1018": [
        {"name": "Metformin",    "dose": "500 mg", "frequency": "BID"},
        {"name": "Lisinopril",   "dose": "10 mg",  "frequency": "QD"},
        {"name": "Atorvastatin", "dose": "20 mg",  "frequency": "QHS"},
        {"name": "Aspirin",      "dose": "81 mg",  "frequency": "QD"},
    ],
    # Perfect reconciliation — discharge meds match these exactly
    "P1019": [
        {"name": "Metformin",    "dose": "500 mg", "frequency": "BID"},
        {"name": "Lisinopril",   "dose": "10 mg",  "frequency": "QD"},
        {"name": "Atorvastatin", "dose": "20 mg",  "frequency": "QHS"},
        {"name": "Aspirin",      "dose": "81 mg",  "frequency": "QD"},
    ],
    # Spanish discharge — EHR orders use the same (Spanish) drug spellings so
    # canonicalization reconciles them exactly against the discharge note.
    "P1020": [
        {"name": "Metformina",   "dose": "500 mg", "frequency": "BID"},
        {"name": "Lisinopril",   "dose": "10 mg",  "frequency": "QD"},
        {"name": "Atorvastatina","dose": "20 mg",  "frequency": "QHS"},
        {"name": "Aspirina",     "dose": "81 mg",  "frequency": "QD"},
    ],
    # Exact match against the Hindi discharge note -> no med reconciliation
    # discrepancies (HITL comes from the unpaid bill + missing data instead).
    "P1021": [
        {"name": "Metformin",    "dose": "500 mg", "frequency": "BID"},
        {"name": "Amlodipine",   "dose": "5 mg",   "frequency": "QD"},
        {"name": "Atorvastatin", "dose": "20 mg",  "frequency": "QHS"},
    ],
    # Meds reconcile exactly (Amoxicilline->amoxicillin, Paracetamol->
    # acetaminophen) so the allergy contradiction is the lone med-related flag.
    "P1022": [
        {"name": "Amoxicillin",  "dose": "500 mg", "frequency": "TID x 7 days"},
        {"name": "Paracetamol",  "dose": "500 mg", "frequency": "q6h PRN"},
    ],
    # Exact match against the English discharge note -> no med reconciliation
    # discrepancies (no omissions, no additions, no high-risk meds).
    "P1023": [
        {"name": "Metformin",    "dose": "500 mg", "frequency": "BID"},
        {"name": "Lisinopril",   "dose": "10 mg",  "frequency": "QD"},
        {"name": "Atorvastatin", "dose": "20 mg",  "frequency": "QHS"},
        {"name": "Aspirin",      "dose": "81 mg",  "frequency": "QD"},
    ],
    # Meds reconcile exactly (Amoxicilline->amoxicillin, Paracetamol->
    # acetaminophen) so the allergy contradiction is the lone med-related flag.
    "P1024": [
        {"name": "Amoxicillin",  "dose": "500 mg", "frequency": "TID x 7 days"},
        {"name": "Paracetamol",  "dose": "500 mg", "frequency": "q6h PRN"},
    ],
}

# ---------------------------------------------------------------------------
# Lab results (those flagged "abnormal" must be reconciled in discharge)
# ---------------------------------------------------------------------------
LABS: dict[str, list[dict]] = {
    "P1001": [
        {"test": "HbA1c", "value": "7.4 %",    "abnormal": True,  "action_in_ehr": "continue Metformin, recheck in 3 months"},
        {"test": "LDL",   "value": "132 mg/dL","abnormal": True,  "action_in_ehr": "start Atorvastatin"},
    ],
    "P1002": [
        {"test": "BNP",    "value": "612 pg/mL", "abnormal": True, "action_in_ehr": "continue diuretics, follow up in 2 weeks"},
        {"test": "Sodium", "value": "134 mmol/L","abnormal": True, "action_in_ehr": "outpatient monitoring"},
    ],
    "P1003": [
        {"test": "Troponin-I", "value": "4.8 ng/mL", "abnormal": True, "action_in_ehr": "serial trended"},
        {"test": "LDL",        "value": "168 mg/dL", "abnormal": True, "action_in_ehr": "high-intensity statin"},
        {"test": "INR",        "value": "1.2",       "abnormal": True, "action_in_ehr": "anticoagulation plan to be confirmed"},
    ],
    "P1004": [
        {"test": "WBC", "value": "14.2 x10^9/L", "abnormal": True, "action_in_ehr": "trending down with antibiotics"},
        {"test": "CRP", "value": "86 mg/L",      "abnormal": True, "action_in_ehr": "trending down with antibiotics"},
    ],
    "P1005": [
        {"test": "Peak flow", "value": "Improved to baseline", "abnormal": False, "action_in_ehr": ""},
    ],
    "P1006": [
        {"test": "Hemoglobin", "value": "9.8 g/dL", "abnormal": True, "action_in_ehr": "oral iron, recheck in 6 weeks"},
    ],
    # Abnormal BNP at 845 with no documented action -> abnormal_lab_unresolved
    "P1007": [
        {"test": "BNP",       "value": "845 pg/mL", "abnormal": True, "action_in_ehr": ""},
        {"test": "Potassium", "value": "3.2 mmol/L","abnormal": True, "action_in_ehr": "oral KCl 20 mEq/day"},
    ],
    "P1008": [],
    "P1009": [
        {"test": "SpO2", "value": "88 % RA",   "abnormal": True, "action_in_ehr": "home oxygen"},
        {"test": "WBC",  "value": "12.5 x10^9","abnormal": True, "action_in_ehr": "antibiotic review"},
    ],
    "P1010": [],
    "P1011": [
        {"test": "WBC", "value": "12.8 x10^9/L", "abnormal": True,
         "action_in_ehr": "trending down with antibiotics; recheck in 1 week"},
        {"test": "CRP", "value": "62 mg/L",      "abnormal": True,
         "action_in_ehr": "trending down with antibiotics"},
    ],
    "P1012": [
        {"test": "Urinalysis", "value": "positive nitrites", "abnormal": True,
         "action_in_ehr": "antibiotic course completed"},
    ],
    "P1013": [],
    "P1014": [],
    "P1015": [],
    "P1016": [],
    "P1017": [],
    # Abnormal labs that are fully resolved (action documented) in discharge
    "P1018": [
        {"test": "HbA1c", "value": "7.1 %",     "abnormal": True, "action_in_ehr": "continue Metformin, recheck in 3 months"},
        {"test": "LDL",   "value": "128 mg/dL", "abnormal": True, "action_in_ehr": "continue statin"},
    ],
    # Abnormal labs that are fully resolved (action documented) in discharge
    "P1019": [
        {"test": "HbA1c", "value": "6.9 %",     "abnormal": True, "action_in_ehr": "continue Metformin, recheck in 3 months"},
        {"test": "LDL",   "value": "118 mg/dL", "abnormal": True, "action_in_ehr": "continue statin"},
    ],
    # All labs within normal limits -> no abnormal-lab reconciliation needed.
    "P1020": [],
    # All labs within normal limits -> no abnormal-lab reconciliation needed.
    "P1021": [
        {"test": "HbA1c",          "value": "6.4 %",      "abnormal": False, "action_in_ehr": ""},
        {"test": "LDL",            "value": "96 mg/dL",   "abnormal": False, "action_in_ehr": ""},
    ],
    # CRP mildly elevated but resolving (recovering pneumonia) -> not flagged
    # as an unresolved abnormal lab.
    "P1022": [
        {"test": "CRP",        "value": "42 mg/L",      "abnormal": False, "action_in_ehr": ""},
        {"test": "Leukocytes", "value": "9.4 x10^9/L",  "abnormal": False, "action_in_ehr": ""},
    ],
    # All labs within normal limits -> no abnormal-lab reconciliation needed.
    "P1023": [
        {"test": "HbA1c",          "value": "5.4 %",     "abnormal": False, "action_in_ehr": ""},
        {"test": "LDL",            "value": "92 mg/dL",  "abnormal": False, "action_in_ehr": ""},
        {"test": "Fasting Glucose","value": "94 mg/dL",  "abnormal": False, "action_in_ehr": ""},
    ],
    # CRP mildly elevated but resolving (recovering pneumonia) -> not flagged
    # as an unresolved abnormal lab; HITL driver is the allergy contradiction.
    "P1024": [
        {"test": "CRP",        "value": "38 mg/L",      "abnormal": False, "action_in_ehr": ""},
        {"test": "Leukocytes", "value": "8.9 x10^9/L",  "abnormal": False, "action_in_ehr": ""},
    ],
}

# ---------------------------------------------------------------------------
# Care plans (the discharge note must include these follow-ups)
# ---------------------------------------------------------------------------
CARE_PLANS: dict[str, dict] = {
    "P1001": {"followup_required": True, "speciality": "Endocrinology", "window_days": 14},
    "P1002": {"followup_required": True, "speciality": "Cardiology",     "window_days": 14},
    "P1003": {"followup_required": True, "speciality": "Cardiology",     "window_days": 7},
    "P1004": {"followup_required": True, "speciality": "PCP",            "window_days": 7},
    "P1005": {"followup_required": True, "speciality": "Pulmonology",    "window_days": 21},
    "P1006": {"followup_required": True, "speciality": "Gastroenterology","window_days": 21},
    "P1007": {"followup_required": True, "speciality": "Cardiology",     "window_days": 7},
    "P1008": {"followup_required": True, "speciality": "PCP",            "window_days": 14},
    "P1009": {"followup_required": True, "speciality": "Pulmonology",    "window_days": 14},
    "P1010": {"followup_required": True, "speciality": "Pulmonology",    "window_days": 21},
    "P1011": {"followup_required": True, "speciality": "PCP",            "window_days": 14},
    "P1012": {"followup_required": True, "speciality": "PCP",            "window_days": 14},
    "P1013": {"followup_required": True, "speciality": "General Surgery","window_days": 14},
    "P1014": {"followup_required": True, "speciality": "PCP",            "window_days": 14},
    "P1015": {"followup_required": True, "speciality": "PCP",            "window_days": 14},
    "P1016": {"followup_required": True, "speciality": "PCP",            "window_days": 14},
    "P1017": {"followup_required": True, "speciality": "PCP",            "window_days": 14},
    "P1018": {"followup_required": True, "speciality": "Endocrinology",  "window_days": 30},
    "P1019": {"followup_required": True, "speciality": "Endocrinology",  "window_days": 30},
    "P1020": {"followup_required": True, "speciality": "Endocrinology",  "window_days": 30},
    # Care plan requires Endocrinology follow-up; the discharge note omits it,
    # so this also surfaces as a followup_missing data gap.
    "P1021": {"followup_required": True, "speciality": "Endocrinology",  "window_days": 30},
    # Follow-up is documented in the discharge note, so no followup_missing —
    # keeps the HITL drivers limited to allergy contradiction + missing demo data.
    "P1022": {"followup_required": True, "speciality": "PCP",            "window_days": 14},
    # Follow-up is documented in the discharge note -> no followup_missing flag.
    "P1023": {"followup_required": True, "speciality": "Endocrinology",  "window_days": 30},
    # Follow-up is documented in the discharge note, so no followup_missing —
    # keeps the HITL drivers limited to allergy contradiction + missing data.
    "P1024": {"followup_required": True, "speciality": "PCP",            "window_days": 14},
}

# ---------------------------------------------------------------------------
# Simulated clinical pathway / guidelines lookup
# ---------------------------------------------------------------------------
GUIDELINES: dict[str, dict] = {
    "E11.9":  {"diagnosis": "Type 2 Diabetes Mellitus",
               "required_followup": "Endocrinology within 30 days",
               "essential_meds": ["Metformin"]},
    "I10":    {"diagnosis": "Hypertension",
               "required_followup": "PCP within 30 days",
               "essential_meds": ["ACE inhibitor or ARB"]},
    "I50.9":  {"diagnosis": "Congestive Heart Failure",
               "required_followup": "Cardiology within 14 days",
               "essential_meds": ["Diuretic", "ACE inhibitor", "Beta-blocker"]},
    "I21.9":  {"diagnosis": "Acute Myocardial Infarction",
               "required_followup": "Cardiology within 7 days",
               "essential_meds": ["Aspirin", "P2Y12 inhibitor", "Statin", "Beta-blocker"]},
    "J45.909":{"diagnosis": "Asthma exacerbation",
               "required_followup": "Pulmonology within 30 days",
               "essential_meds": ["SABA", "ICS"]},
    "J18.9":  {"diagnosis": "Pneumonia",
               "required_followup": "PCP within 14 days",
               "essential_meds": ["Antibiotic"]},
    "J20.9":  {"diagnosis": "Acute bronchitis",
               "required_followup": "PCP within 14 days",
               "essential_meds": []},
    "J44.9":  {"diagnosis": "COPD exacerbation",
               "required_followup": "Pulmonology within 14 days",
               "essential_meds": ["LAMA", "SABA", "Steroid burst"]},
    "N39.0":  {"diagnosis": "Urinary tract infection",
               "required_followup": "PCP within 14 days",
               "essential_meds": ["Antibiotic"]},
    "K35.80": {"diagnosis": "Acute appendicitis",
               "required_followup": "Surgery within 14 days",
               "essential_meds": ["Antibiotic", "Analgesic"]},
    "K52.9":  {"diagnosis": "Acute gastroenteritis",
               "required_followup": "PCP within 14 days",
               "essential_meds": ["Antiemetic", "Oral rehydration"]},
    "A01.0":  {"diagnosis": "Typhoid (enteric) fever",
               "required_followup": "PCP within 14 days",
               "essential_meds": ["Antibiotic"]},
    "N10":    {"diagnosis": "Acute pyelonephritis",
               "required_followup": "PCP within 14 days",
               "essential_meds": ["Antibiotic"]},
}
