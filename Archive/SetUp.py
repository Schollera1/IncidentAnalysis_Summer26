# Set-Up File, do this first.
import re

excel_file_name = "XO_incidents_RCA_merchants.xlsx"
sheet_name = "Data - Updated"
ticket_number = "number"
ticket_short_description = "short_description"
ticket_taxonomy_final = "Root Cause Category"
ticket_taxonomy_original = "Root Cause Category (original)"
ticket_reassigned = "Reassigned Root Cause Category"
ticket_merchant = "Merchant / Partner"
ticket_month_year = "sys_created_on"
ticket_impact = "impact"
ticket_urgency = "urgency"
ticket_priority = "priority"
ticket_executive_summary = "u_executive_summary"
ticket_reassignment_rationale = "Reassigned Root Cause Category"
ticket_incident_details = "u_provide_incident_details"
ticket_product = "u_app_product_service"
ticket_opened = "opened_at"
ticket_updated = "sys_updated_on"
# text columns that WILL be fed for the model to analyze/classify tickets
model_text_tickets_data = [
    ticket_short_description,
    ticket_incident_details,
    ticket_executive_summary,
]
# text columns that will NOT be fed for the model to analyze/classify tickets
leaky_columns = {
    ticket_taxonomy_final,
    ticket_taxonomy_original,
    ticket_reassigned,
    "Reassignment Rationale",
    "Merchant Reassignment Rationale",
    "RCA Note (If Report wasn't Present)",
}

# L2 (New) Taxonomies made by Sohan, just the flat list, kept for reference
new_l2_labels = {
    "Unclassified / Other",
    "Merchant-side Misconfiguration / Account Issue",
    "Internal Merchant Misconfiguration / Account Issue",
    "Configuration Regression",
    "Code Deployment Regression",
    "Merchant-side Vendor Failure",
    "Payment Partner Failure",
    "Compute / Network / Service Outage",
    "Conversion / Volume Variance",
    "API Contract / Validation Bug",
    "Security / Abuse / Bot Attack",
    "3DS / Authentication / Login Failure",
    "Planned Maintenance / Migration Impact",
    "Risk / Fraud Engine Over-trigger",
    "JS SDK / Button Rendering Issue",
    "Webhook / Event Notification Bug",
}

# each new L2 label and the new L1 family it belongs to
# (labels that point to themselves are already their own top-level family)
l2_to_l1 = {
    "Compute / Network / Service Outage": "Infrastructure / Service Availability",
    "Planned Maintenance / Migration Impact": "Infrastructure / Service Availability",
    "Merchant-side Misconfiguration / Account Issue": "Merchant Misconfiguration / Account Issue",
    "Internal Merchant Misconfiguration / Account Issue": "Merchant Misconfiguration / Account Issue",
    "Configuration Regression": "Code / Config Deployment Regression",
    "Code Deployment Regression": "Code / Config Deployment Regression",
    "Merchant-side Vendor Failure": "Third-party / External Dependency Failure",
    "Payment Partner Failure": "Third-party / External Dependency Failure",
    "Conversion / Volume Variance": "Conversion / Volume Variance",
    "API Contract / Validation Bug": "API Contract / Validation Bug",
    "Security / Abuse / Bot Attack": "Security / Abuse / Bot Attack",
    "3DS / Authentication / Login Failure": "3DS / Authentication / Login Failure",
    "Risk / Fraud Engine Over-trigger": "Risk / Fraud Engine Over-trigger",
    "JS SDK / Button Rendering Issue": "JS SDK / Button Rendering Issue",
    "Webhook / Event Notification Bug": "Webhook / Event Notification Bug",
}
# old labels that just got renamed, map them straight to the new L2
label_renames = {
    "Unexplained Conversion / UX Drop-off": "Conversion / Volume Variance",
}

# broad old labels that cover more than one new L2, so a human has to split them later
legacy_parent_splits = {
    "Infrastructure / Service Availability": (
        "Compute / Network / Service Outage",
        "Planned Maintenance / Migration Impact",
    ),
    "Merchant Misconfiguration / Account Issue": (
        "Merchant-side Misconfiguration / Account Issue",
        "Internal Merchant Misconfiguration / Account Issue",
    ),
    "Code / Config Deployment Regression": (
        "Configuration Regression",
        "Code Deployment Regression",
    ),
    "Third-party / External Dependency Failure": (
        "Payment Partner Failure",
        "Merchant-side Vendor Failure",
    ),
}
# light data cleaning with exclusions/abstaining labels, setting minimum threshold for training data cluster, and P2 priority, and human-reviewed tickets
exclude_labels = {"Test / Admin (exclude)"}
abstain_labels = {"Unclassified / Other"}
min_support_for_l2 = 15
restrict_to_priority = "P2 - High"  # set to None to keep every priority
human_confirmed_start = "2025-10-01"
human_confirmed_end = "2026-05-31"


# Makes clean strings
def combine_ticket_text(ticket_cells, text_columns=model_text_tickets_data):
    text_analyzed = []
    for column_name in text_columns:
        cell = ticket_cells.get(column_name)
        if cell is not None and str(cell).strip() and str(cell).lower() != "nan":
            text_analyzed.append(str(cell).strip())
    combined_text = "\n ".join(text_analyzed)
    combined_text = re.sub(r"_x000d_", " ", combined_text, flags=re.IGNORECASE)
    combined_text = combined_text.replace("\r", " ")
    combined_text = re.sub(r"https?://\S+", " ", combined_text)
    combined_text = re.sub(r"\s+", " ", combined_text).strip()
    return combined_text


def combine_ticket_text_masked(ticket_cells, text_columns=model_text_tickets_data):
    combined_text = combine_ticket_text(ticket_cells, text_columns)
    merchant = ticket_cells.get(ticket_merchant)
    merchant = str(merchant).strip() if merchant is not None else ""
    if merchant and merchant.lower() not in ("nan", "no specific merchant"):
        combined_text = re.sub(
            re.escape(merchant), "[MERCHANT]", combined_text, flags=re.IGNORECASE
        )
    return combined_text
