"""Entity extraction using spaCy NER + regex patterns for telecom entities."""
import re
import spacy

nlp = spacy.load("en_core_web_sm")

# Regex patterns for telecom entities
SPEC_PATTERN = re.compile(r"(?:3GPP\s+)?(?:TS|TR)\s*(\d{2}\.\d{3}(?:-\d+)?)", re.IGNORECASE)
RELEASE_PATTERN = re.compile(r"Release\s+(\d{1,2})", re.IGNORECASE)
TECH_CONCEPTS = [
    "5G NR", "5G Core", "5GC", "5GS", "LTE", "E-UTRAN", "EPC",
    "IMS", "IMS-ALG", "IMS-AGW", "VoNR", "VoLTE",
    "O-RAN", "ORAN", "RIC", "Near-RT RIC", "Non-RT RIC",
    "NWDAF", "PCF", "AMF", "SMF", "UPF", "NEF", "NRF", "AUSF", "UDM",
    "Network Slicing", "URLLC", "eMBB", "mMTC",
    "MIMO", "Massive MIMO", "Beamforming",
    "NTN", "Non-Terrestrial", "Satellite",
    "MEC", "Edge Computing", "EDGEAPP",
    "V2X", "Sidelink", "PC5", "ProSe",
    "MCPTT", "MCData", "MCVideo", "Mission Critical",
    "ECN", "QoS", "SBA", "SBI",
    "SEAL", "IAB", "DSS", "RedCap",
    "HARQ", "LDPC", "Polar", "OFDM",
    "F1AP", "XnAP", "NGAP", "E2",
    "gNB", "eNB", "CU", "DU", "RU",
    "PDU Session", "QoS Flow", "DRB",
    "SNPN", "PNI-NPN", "CAG",
]

RELATION_KEYWORDS = {
    "references": ["references", "refers to", "see also", "as defined in", "specified in"],
    "depends_on": ["depends on", "requires", "relies on", "based on", "built upon"],
    "defines": ["defines", "specifies", "describes", "introduces", "establishes"],
    "extends": ["extends", "enhances", "builds upon", "evolution of", "upgrade"],
    "part_of": ["part of", "included in", "within", "subset of", "component of"],
    "related_to": ["related to", "associated with", "connected to", "relevant to"],
}


def extract_entities(text):
    """Extract telecom entities from text."""
    entities = []

    # Extract spec numbers
    for match in SPEC_PATTERN.finditer(text):
        spec_num = match.group(1)
        prefix = "TS" if "TS" in match.group(0).upper() else "TR"
        entities.append({
            "name": f"{prefix} {spec_num}",
            "type": "spec",
            "start": match.start(),
            "end": match.end(),
        })

    # Extract release numbers
    for match in RELEASE_PATTERN.finditer(text):
        entities.append({
            "name": f"Release {match.group(1)}",
            "type": "release",
            "start": match.start(),
            "end": match.end(),
        })

    # Extract technology concepts
    text_upper = text.upper()
    for concept in TECH_CONCEPTS:
        idx = text_upper.find(concept.upper())
        if idx != -1:
            entities.append({
                "name": concept,
                "type": "concept",
                "start": idx,
                "end": idx + len(concept),
            })

    # Deduplicate by name
    seen = set()
    unique = []
    for e in entities:
        if e["name"] not in seen:
            seen.add(e["name"])
            unique.append(e)

    return unique


def extract_relations(text, entities):
    """Extract relationships between entities based on proximity and keywords."""
    relations = []
    entity_names = [e["name"] for e in entities]

    sentences = text.split(".")
    for sentence in sentences:
        # Find entities in this sentence
        sent_entities = [e for e in entities if e["name"].lower() in sentence.lower()]
        if len(sent_entities) < 2:
            continue

        # Determine relation type from keywords
        rel_type = "related_to"
        confidence = 0.5
        for rtype, keywords in RELATION_KEYWORDS.items():
            for kw in keywords:
                if kw in sentence.lower():
                    rel_type = rtype
                    confidence = 0.8
                    break

        # Create edges between entity pairs in the sentence
        for i in range(len(sent_entities)):
            for j in range(i + 1, len(sent_entities)):
                relations.append({
                    "source": sent_entities[i]["name"],
                    "target": sent_entities[j]["name"],
                    "relation": rel_type,
                    "confidence": confidence,
                    "context": sentence.strip()[:200],
                })

    return relations


def extract_from_response(response_text):
    """Extract entities and relations from LLM response for graph update."""
    entities = extract_entities(response_text)
    relations = extract_relations(response_text, entities)
    return entities, relations
