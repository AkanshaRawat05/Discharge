"""
dashboard/uploads.py
====================

Lets a reviewer drop new clinical documents (discharge report, lab report,
hospital bill) into the hospital's incoming folder straight from the console.

Nothing downstream changes: the files are written into the very same
`Data/incoming/{doctor_reports,lab_reports,bills}` folders the dataset already
lives in, under a `P####_…` filename, so `scan_incoming()` discovers the patient
and `find_patient_files()` picks the upload up exactly like a file the hospital's
own drop had produced.  **Process this patient** then runs the normal pipeline.

Two rules are enforced on the way in, both borrowed from the MCP Roots contract
the Clinical Watcher tool works under:

* the target must resolve inside the declared input root (`resolve_within_root`)
* the filename must carry a patient id the loader's `PATIENT_ID_RE` can read,
  so one is prefixed when the uploaded name lacks it
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from discharge_ai.common.doc_loader import (
    OCR_SIDECAR_MARKER,
    SUPPORTED_SUFFIXES,
    extract_patient_id,
    find_patient_files,
    resolve_within_root,
)
from discharge_ai.common.schemas import DocType
from discharge_ai.settings import settings

#  Document type → (config path key, label shown in the uploader)
DOC_TYPE_TARGETS: dict[str, tuple[str, str]] = {
    DocType.DISCHARGE_REPORT.value: ("doctor_reports", "Discharge / doctor report"),
    DocType.LAB_REPORT.value: ("lab_reports", "Lab report"),
    DocType.BILL.value: ("bills", "Hospital bill"),
}

#  What the uploader hands to Streamlit's `type=` filter (no leading dots).
ACCEPTED_EXTENSIONS = sorted(suffix.lstrip(".") for suffix in SUPPORTED_SUFFIXES)

#  A patient id the rest of the system can find again: `P` + exactly 4 digits,
#  which is what `doc_loader.PATIENT_ID_RE` looks for in a filename.
PATIENT_ID_INPUT_RE = re.compile(r"^P\d{4}$", re.IGNORECASE)

#  Everything else in a filename is replaced — the name reaches the filesystem.
UNSAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class SavedUpload:
    """One stored file, and whether it landed on top of an existing one."""

    doc_type: str
    path: Path
    original_name: str
    replaced: bool


# --------------------------------------------------------------------------- #
#  Naming and storage
# --------------------------------------------------------------------------- #
def normalise_patient_id(raw: str | None) -> str | None:
    """`  p1025 ` → `P1025`; anything that is not `P####` → None."""
    candidate = (raw or "").strip()
    return candidate.upper() if PATIENT_ID_INPUT_RE.match(candidate) else None


def safe_filename(original_name: str, patient_id: str) -> str:
    """A filesystem-safe name that carries the patient id.

    `Path(...).name` drops any directory component a browser may have sent, the
    remainder is reduced to `[A-Za-z0-9._-]`, and the patient id is prefixed
    unless the name already contains that same id.
    """
    name = UNSAFE_CHARS_RE.sub("_", Path(original_name).name).strip("._") or "document"

    if extract_patient_id(Path(name)) != patient_id:
        name = f"{patient_id}_{name}"
    return name


def save_upload(
    data: bytes,
    original_name: str,
    doc_type: str,
    patient_id: str,
) -> SavedUpload:
    """Write one uploaded document into its incoming folder.

    Raises `ValueError` for an unsupported format and `RootAccessError` if the
    resolved target would sit outside the declared MCP input root.
    """
    path_key, _label = DOC_TYPE_TARGETS[doc_type]

    suffix = Path(original_name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file format '{suffix or original_name}'. Accepted: "
            + ", ".join(f".{ext}" for ext in ACCEPTED_EXTENSIONS)
        )

    folder = settings.path(path_key)
    filename = safe_filename(original_name, patient_id)

    #  Same guard the Clinical Watcher tool applies: prove the write stays
    #  inside `Data/incoming` before touching the disk.
    target = resolve_within_root(folder / filename, settings.path("input_root"))

    replaced = target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)

    return SavedUpload(
        doc_type=doc_type,
        path=target,
        original_name=original_name,
        replaced=replaced,
    )


# --------------------------------------------------------------------------- #
#  UI panel  (rendered at the top of view 1)
# --------------------------------------------------------------------------- #
def render_upload_panel() -> None:
    """Expander that accepts new documents for a patient and stores them.

    On success the new patient is queued as the sidebar selection (see
    `common.patient_selector`) and the page reruns, so the freshly uploaded
    record is the one on screen when the reviewer presses *Process*.
    """
    patients_exist = bool(st.session_state.get("patient_id"))
    #  A save ends in `st.rerun()`, so the confirmations are carried across that
    #  rerun instead of being written straight to a page about to be discarded.
    flash: list[tuple[str, str]] = st.session_state.pop("upload_flash", [])

    with st.expander("📤 Upload new documents", expanded=bool(flash) or not patients_exist):
        for level, message in flash:
            getattr(st, level)(message)

        st.caption(
            "Drop a patient's discharge report, lab report and hospital bill here. "
            f"They are stored under `{settings.path('input_root')}` alongside the "
            "records the hospital already sent, and behave identically from there — "
            "select the patient and press **Process this patient**."
        )

        #  The key carries the selected patient so the box is prefilled afresh
        #  for each one: Streamlit ignores `value` once a keyed widget has state,
        #  and a box still showing the previous patient's id invites a
        #  misfiled upload.
        default_id = st.session_state.get("patient_id") or ""
        patient_id_raw = st.text_input(
            "Patient ID",
            value=default_id,
            placeholder="P1025",
            help="Format `P` followed by four digits — this is how every part of "
                 "the system identifies the case.",
            key=f"upload-patient-id-{default_id or 'new'}",
        )

        columns = st.columns(len(DOC_TYPE_TARGETS))
        pending: list[tuple[str, object]] = []
        for column, (doc_type, (_key, label)) in zip(columns, DOC_TYPE_TARGETS.items()):
            with column:
                files = st.file_uploader(
                    label,
                    type=ACCEPTED_EXTENSIONS,
                    accept_multiple_files=True,
                    key=f"upload-{doc_type}",
                )
                pending.extend((doc_type, file) for file in files or [])

        st.caption(
            "Accepted: "
            + ", ".join(f"`.{ext}`" for ext in ACCEPTED_EXTENSIONS)
            + ". Scans and image PDFs need OCR — upload the matching "
            f"`<name>{OCR_SIDECAR_MARKER}` text file next to them (any of the three "
            "boxes will do), or install Tesseract."
        )

        if not st.button("Save to incoming folder", type="primary",
                         disabled=not pending, key="upload-save"):
            return

        patient_id = normalise_patient_id(patient_id_raw)
        if not patient_id:
            st.error(
                "Enter a patient ID as `P` plus four digits (for example `P1025`) "
                "before saving — the pipeline finds documents by that id."
            )
            return

        _store(pending, patient_id)


def _repo_relative(path: Path) -> Path:
    """`Data/incoming/bills/P1025_bill.json` when the root allows it."""
    try:
        return path.relative_to(settings.root)
    except ValueError:          # input_root configured outside the repository
        return path


def _store(pending: list[tuple[str, object]], patient_id: str) -> None:
    """Write every queued upload, then report what the loader will now use."""
    saved: list[SavedUpload] = []
    flash: list[tuple[str, str]] = []

    for doc_type, file in pending:
        try:
            upload = save_upload(file.getvalue(), file.name, doc_type, patient_id)
        except Exception as exc:  # noqa: BLE001 — surfaced to the reviewer below
            flash.append(("error", f"`{file.name}` — {exc}"))
            continue

        saved.append(upload)
        flash.append((
            "success",
            f"{DOC_TYPE_TARGETS[upload.doc_type][1]}: "
            f"{'replaced' if upload.replaced else 'saved'} "
            f"`{_repo_relative(upload.path)}`",
        ))

    if not saved:
        #  Nothing was written, so the page is not about to be rerun: the
        #  failures have to be rendered here or they are never seen.
        for level, message in flash:
            getattr(st, level)(message)
        return

    #  A patient can end up with several files of one type (a JSON and a scan,
    #  say).  `find_patient_files()` keeps the best-readable one, so show which
    #  file the pipeline will actually read rather than let the reviewer assume
    #  it is always the newest upload.
    in_use = find_patient_files(patient_id)
    if in_use:
        flash.append((
            "info",
            f"**{patient_id}** will be processed from: "
            + " · ".join(
                f"{DOC_TYPE_TARGETS.get(doc_type, (None, doc_type))[1]} `{path.name}`"
                for doc_type, path in sorted(in_use.items())
            ),
        ))
    st.session_state["upload_flash"] = flash

    #  Clear the uploader widgets (deleting a widget key is the supported reset;
    #  assigning to one after it rendered is not) and hand the sidebar its new
    #  selection.
    for doc_type in DOC_TYPE_TARGETS:
        if f"upload-{doc_type}" in st.session_state:
            del st.session_state[f"upload-{doc_type}"]
    st.session_state["pending_patient_id"] = patient_id
    st.session_state["force_case_refresh"] = True
    st.rerun()
