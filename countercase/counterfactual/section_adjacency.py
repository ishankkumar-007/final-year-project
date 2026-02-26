"""Section adjacency map for legal section perturbation.

Maps normalised section strings to their legally adjacent sections.
Adjacent sections are those in the same chapter, or sections commonly
cited together in Indian case law, making substitution perturbations
plausible.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Adjacency map
# -------------------------------------------------------------------
# Keys and values are normalised section strings matching the format
# used in FactSheet.sections_cited.

ADJACENCY_MAP: dict[str, list[str]] = {
    # ---------------------------------------------------------------
    # Indian Penal Code (IPC)
    # ---------------------------------------------------------------
    # Murder / culpable homicide cluster
    "IPC-299": ["IPC-300", "IPC-304", "IPC-304A"],
    "IPC-300": ["IPC-299", "IPC-302", "IPC-304"],
    "IPC-302": ["IPC-300", "IPC-304", "IPC-304A", "IPC-307"],
    "IPC-304": ["IPC-302", "IPC-304A", "IPC-304B", "IPC-300"],
    "IPC-304A": ["IPC-304", "IPC-302", "IPC-338"],
    "IPC-304B": ["IPC-304", "IPC-498A", "IPC-306"],
    "IPC-306": ["IPC-304B", "IPC-304", "IPC-498A", "IPC-302"],
    "IPC-307": ["IPC-302", "IPC-308", "IPC-326"],

    # Hurt / grievous hurt
    "IPC-319": ["IPC-320", "IPC-321", "IPC-323"],
    "IPC-320": ["IPC-319", "IPC-322", "IPC-325", "IPC-326"],
    "IPC-323": ["IPC-319", "IPC-324", "IPC-325"],
    "IPC-324": ["IPC-323", "IPC-325", "IPC-326"],
    "IPC-325": ["IPC-323", "IPC-324", "IPC-326"],
    "IPC-326": ["IPC-325", "IPC-307", "IPC-324"],

    # Sexual offences
    "IPC-376": ["IPC-376A", "IPC-376B", "IPC-376C", "IPC-376D", "IPC-354"],
    "IPC-376A": ["IPC-376", "IPC-376B", "IPC-302"],
    "IPC-376B": ["IPC-376", "IPC-376A", "IPC-376C"],
    "IPC-376C": ["IPC-376", "IPC-376B", "IPC-376D"],
    "IPC-376D": ["IPC-376", "IPC-376C"],
    "IPC-354": ["IPC-376", "IPC-354A", "IPC-354B", "IPC-509"],
    "IPC-354A": ["IPC-354", "IPC-354B"],
    "IPC-354B": ["IPC-354", "IPC-354A", "IPC-376"],

    # Cheating / fraud
    "IPC-415": ["IPC-420", "IPC-417"],
    "IPC-417": ["IPC-415", "IPC-418"],
    "IPC-420": ["IPC-415", "IPC-406", "IPC-467", "IPC-468"],

    # Criminal breach of trust
    "IPC-405": ["IPC-406", "IPC-408", "IPC-409"],
    "IPC-406": ["IPC-405", "IPC-420"],
    "IPC-408": ["IPC-405", "IPC-409"],
    "IPC-409": ["IPC-405", "IPC-408"],

    # Dowry / cruelty
    "IPC-498A": ["IPC-304B", "IPC-306", "IPC-34"],

    # Common intention / conspiracy / unlawful assembly
    "IPC-34": ["IPC-120B", "IPC-149"],
    "IPC-120B": ["IPC-34", "IPC-149"],
    "IPC-149": ["IPC-34", "IPC-120B", "IPC-148"],
    "IPC-148": ["IPC-149", "IPC-147"],
    "IPC-147": ["IPC-148", "IPC-149"],

    # Forgery
    "IPC-463": ["IPC-464", "IPC-467", "IPC-468"],
    "IPC-467": ["IPC-463", "IPC-468", "IPC-471"],
    "IPC-468": ["IPC-467", "IPC-420", "IPC-471"],
    "IPC-471": ["IPC-467", "IPC-468"],

    # Kidnapping / abduction
    "IPC-359": ["IPC-360", "IPC-361", "IPC-362", "IPC-363"],
    "IPC-363": ["IPC-359", "IPC-364", "IPC-366"],
    "IPC-364": ["IPC-363", "IPC-364A"],
    "IPC-364A": ["IPC-364", "IPC-302"],

    # Robbery / dacoity
    "IPC-392": ["IPC-393", "IPC-394", "IPC-395"],
    "IPC-395": ["IPC-396", "IPC-397", "IPC-392"],
    "IPC-397": ["IPC-395", "IPC-396"],

    # Defamation
    "IPC-499": ["IPC-500"],
    "IPC-500": ["IPC-499"],

    # Miscellaneous
    "IPC-338": ["IPC-304A", "IPC-337"],
    "IPC-337": ["IPC-338", "IPC-304A"],
    "IPC-509": ["IPC-354", "IPC-354A"],

    # ---------------------------------------------------------------
    # Code of Civil Procedure (CPC)
    # ---------------------------------------------------------------
    "CPC-Section-9": ["CPC-Section-10", "CPC-Section-11"],
    "CPC-Section-10": ["CPC-Section-9", "CPC-Section-11"],
    "CPC-Section-11": ["CPC-Section-9", "CPC-Section-10", "CPC-Section-12"],
    "CPC-Section-12": ["CPC-Section-11"],
    "CPC-Section-151": ["CPC-Section-9", "CPC-Order-39"],
    "CPC-Order-7-Rule-11": ["CPC-Section-9", "CPC-Order-7-Rule-10"],
    "CPC-Order-7-Rule-10": ["CPC-Order-7-Rule-11"],
    "CPC-Order-39": ["CPC-Order-39-Rule-1", "CPC-Order-39-Rule-2", "CPC-Section-151"],
    "CPC-Order-39-Rule-1": ["CPC-Order-39-Rule-2", "CPC-Order-39"],
    "CPC-Order-39-Rule-2": ["CPC-Order-39-Rule-1", "CPC-Order-39"],

    # ---------------------------------------------------------------
    # Constitution of India
    # ---------------------------------------------------------------
    "Constitution-Article-14": [
        "Constitution-Article-19",
        "Constitution-Article-21",
        "Constitution-Article-15",
        "Constitution-Article-16",
    ],
    "Constitution-Article-15": [
        "Constitution-Article-14",
        "Constitution-Article-16",
    ],
    "Constitution-Article-16": [
        "Constitution-Article-14",
        "Constitution-Article-15",
    ],
    "Constitution-Article-19": [
        "Constitution-Article-14",
        "Constitution-Article-21",
        "Constitution-Article-19A",
    ],
    "Constitution-Article-21": [
        "Constitution-Article-14",
        "Constitution-Article-19",
        "Constitution-Article-21A",
        "Constitution-Article-22",
    ],
    "Constitution-Article-21A": ["Constitution-Article-21"],
    "Constitution-Article-22": ["Constitution-Article-21"],
    "Constitution-Article-32": [
        "Constitution-Article-226",
        "Constitution-Article-136",
    ],
    "Constitution-Article-136": [
        "Constitution-Article-32",
        "Constitution-Article-226",
        "Constitution-Article-227",
    ],
    "Constitution-Article-226": [
        "Constitution-Article-32",
        "Constitution-Article-136",
        "Constitution-Article-227",
    ],
    "Constitution-Article-227": [
        "Constitution-Article-226",
        "Constitution-Article-136",
    ],
    "Constitution-Article-300A": [
        "Constitution-Article-14",
        "Constitution-Article-21",
    ],
    "Constitution-Article-368": [
        "Constitution-Article-13",
        "Constitution-Article-32",
    ],
    "Constitution-Article-13": ["Constitution-Article-368"],

    # ---------------------------------------------------------------
    # Indian Evidence Act
    # ---------------------------------------------------------------
    "Evidence-Section-3": ["Evidence-Section-4", "Evidence-Section-45"],
    "Evidence-Section-4": ["Evidence-Section-3"],
    "Evidence-Section-24": ["Evidence-Section-25", "Evidence-Section-26"],
    "Evidence-Section-25": [
        "Evidence-Section-24",
        "Evidence-Section-26",
        "Evidence-Section-27",
    ],
    "Evidence-Section-26": ["Evidence-Section-25", "Evidence-Section-27"],
    "Evidence-Section-27": [
        "Evidence-Section-25",
        "Evidence-Section-26",
        "Evidence-Section-32",
    ],
    "Evidence-Section-32": [
        "Evidence-Section-27",
        "Evidence-Section-33",
    ],
    "Evidence-Section-33": ["Evidence-Section-32"],
    "Evidence-Section-45": [
        "Evidence-Section-3",
        "Evidence-Section-46",
        "Evidence-Section-47",
    ],
    "Evidence-Section-46": ["Evidence-Section-45"],
    "Evidence-Section-47": ["Evidence-Section-45"],
    "Evidence-Section-65B": ["Evidence-Section-65A", "Evidence-Section-65"],
    "Evidence-Section-65A": ["Evidence-Section-65B"],
    "Evidence-Section-65": ["Evidence-Section-65B"],

    # ---------------------------------------------------------------
    # Code of Criminal Procedure (CrPC)
    # ---------------------------------------------------------------
    "CrPC-Section-154": ["CrPC-Section-155", "CrPC-Section-156"],
    "CrPC-Section-155": ["CrPC-Section-154"],
    "CrPC-Section-156": ["CrPC-Section-154", "CrPC-Section-190"],
    "CrPC-Section-190": ["CrPC-Section-156", "CrPC-Section-200"],
    "CrPC-Section-200": ["CrPC-Section-190", "CrPC-Section-202"],
    "CrPC-Section-202": ["CrPC-Section-200", "CrPC-Section-203"],
    "CrPC-Section-203": ["CrPC-Section-202", "CrPC-Section-204"],
    "CrPC-Section-204": ["CrPC-Section-203"],
    "CrPC-Section-438": ["CrPC-Section-439", "CrPC-Section-437"],
    "CrPC-Section-437": ["CrPC-Section-438", "CrPC-Section-439"],
    "CrPC-Section-439": ["CrPC-Section-438", "CrPC-Section-437"],
    "CrPC-Section-482": ["Constitution-Article-226", "CrPC-Section-397"],
    "CrPC-Section-397": ["CrPC-Section-482", "CrPC-Section-401"],
    "CrPC-Section-401": ["CrPC-Section-397"],

    # ---------------------------------------------------------------
    # POCSO
    # ---------------------------------------------------------------
    "POCSO-Section-3": ["POCSO-Section-4", "POCSO-Section-5"],
    "POCSO-Section-4": ["POCSO-Section-3", "POCSO-Section-6"],
    "POCSO-Section-5": ["POCSO-Section-3", "POCSO-Section-6"],
    "POCSO-Section-6": ["POCSO-Section-4", "POCSO-Section-5"],

    # ---------------------------------------------------------------
    # NDPS
    # ---------------------------------------------------------------
    "NDPS-Section-15": ["NDPS-Section-18", "NDPS-Section-20"],
    "NDPS-Section-18": ["NDPS-Section-15", "NDPS-Section-20"],
    "NDPS-Section-20": ["NDPS-Section-15", "NDPS-Section-18", "NDPS-Section-21"],
    "NDPS-Section-21": ["NDPS-Section-20", "NDPS-Section-22"],
    "NDPS-Section-22": ["NDPS-Section-21"],
    "NDPS-Section-37": ["NDPS-Section-36A"],
    "NDPS-Section-36A": ["NDPS-Section-37"],
}


def get_adjacent_sections(section: str) -> list[str]:
    """Return adjacent sections for a given normalised section string.

    Tries exact match first, then tries common normalisations
    (strip whitespace, upper-case, replace spaces with hyphens).

    Args:
        section: Normalised section string (e.g. ``"IPC-302"``).

    Returns:
        List of adjacent section strings (may be empty).
    """
    # Exact match
    if section in ADJACENCY_MAP:
        return list(ADJACENCY_MAP[section])

    # Normalised match
    key = section.strip().replace(" ", "-")
    if key in ADJACENCY_MAP:
        return list(ADJACENCY_MAP[key])

    # Upper-case normalisation
    key_upper = key.upper()
    for k, v in ADJACENCY_MAP.items():
        if k.upper() == key_upper:
            return list(v)

    logger.debug("No adjacent sections found for '%s'", section)
    return []
