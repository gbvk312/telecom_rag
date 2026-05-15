"""NetworkX graph construction and Pyvis interactive rendering."""
import json
import networkx as nx
from pyvis.network import Network
from config import GRAPH_OUTPUT_PATH, GRAPH_JSON_PATH

# Color scheme for node types
NODE_COLORS = {
    "spec": "#4e9af1",      # Blue
    "release": "#57cc99",   # Green
    "concept": "#f4a261",   # Orange
    "paper": "#c77dff",     # Purple
    "active": "#e63946",    # Red (query highlight)
}

NODE_SIZES = {
    "spec": 20,
    "release": 25,
    "concept": 18,
    "paper": 16,
}

# Seed data for initial graph
SEED_EDGES = [
    ("TS 23.501", "TS 38.300", "references", "5GS Architecture references NR Overall Description"),
    ("TS 38.300", "Release 15", "part_of", "NR Overall Description - Release 15"),
    ("TS 23.334", "IMS-ALG", "defines", "IMS-ALG/IMS-AGW Interface"),
    ("IMS-ALG", "IMS-AGW", "depends_on", "IMS Subsystem dependency"),
    ("TS 23.501", "5G Core", "defines", "System Architecture for 5GS"),
    ("5G Core", "O-RAN", "related_to", "Disaggregated RAN architecture"),
    ("O-RAN", "Release 16", "part_of", "O-RAN Alliance specifications"),
    ("Release 16", "Release 18", "extends", "Feature Enhancement path"),
    ("TS 26.114", "ECN", "defines", "Multimedia Telephony ECN support"),
    ("ECN", "IMS-AGW", "depends_on", "Congestion Control mechanism"),
    ("TS 23.501", "Network Slicing", "defines", "5G Network Slicing architecture"),
    ("TS 23.502", "TS 23.501", "references", "5GS Procedures reference Architecture"),
    ("TS 23.503", "PCF", "defines", "Policy Control Function"),
    ("TS 23.288", "NWDAF", "defines", "Network Data Analytics Function"),
    ("NWDAF", "5G Core", "part_of", "Analytics NF in 5GC"),
    ("TS 38.473", "F1AP", "defines", "F1 Application Protocol"),
    ("TS 38.423", "XnAP", "defines", "Xn Application Protocol"),
    ("TS 38.413", "NGAP", "defines", "NG Application Protocol"),
    ("TS 23.558", "EDGEAPP", "defines", "Edge Application Architecture"),
    ("EDGEAPP", "MEC", "related_to", "Multi-access Edge Computing"),
    ("TS 23.379", "MCPTT", "defines", "Mission Critical PTT"),
    ("TS 23.280", "Mission Critical", "defines", "MC Services Common Architecture"),
    ("TS 23.434", "SEAL", "defines", "Service Enabler Architecture Layer"),
    ("Release 15", "Release 16", "extends", "5G Phase 1 to Phase 2"),
    ("Release 17", "Release 18", "extends", "NTN to 5G-Advanced"),
    ("Release 16", "Release 17", "extends", "URLLC to RedCap"),
    ("TS 36.300", "LTE", "defines", "E-UTRAN Overall Description"),
    ("LTE", "5G NR", "related_to", "4G to 5G evolution"),
    ("5G NR", "TS 38.300", "defines", "NR air interface"),
    ("V2X", "Sidelink", "depends_on", "V2X uses PC5 sidelink"),
    ("TS 23.287", "V2X", "defines", "V2X Application Architecture"),
    ("NTN", "Release 17", "part_of", "Non-Terrestrial Networks"),
    ("Massive MIMO", "5G NR", "part_of", "Key 5G NR technology"),
    ("IAB", "Release 16", "part_of", "Integrated Access and Backhaul"),
    # Whitepapers
    ("Nokia: AI/ML in RAN", "O-RAN", "related_to", "AI in Network Architecture"),
    ("Nokia: AI/ML in RAN", "NWDAF", "related_to", "AI/ML analytics"),
    ("Nokia: AI/ML in RAN", "Release 18", "related_to", "5G-Advanced AI features"),
    ("Ericsson: 5G NR Architecture", "5G NR", "related_to", "NR deployment strategies"),
    ("Ericsson: 5G NR Architecture", "TS 23.501", "references", "5GS Architecture reference"),
    ("Ericsson: IMS in 5G", "IMS-ALG", "related_to", "IMS evolution"),
    ("Ericsson: IMS in 5G", "TS 26.114", "references", "Multimedia Telephony"),
    ("Ericsson: Edge Computing", "EDGEAPP", "related_to", "MEC architecture"),
    ("Ericsson: Edge Computing", "TS 23.558", "references", "Edge App Architecture"),
    ("Ericsson: Mission Critical", "MCPTT", "related_to", "MC services"),
    ("GSMA: Network Slicing", "Network Slicing", "related_to", "Slicing business models"),
    ("GSMA: Network Slicing", "TS 23.501", "references", "5GS slicing architecture"),
    ("GSMA: Private 5G", "5G Core", "related_to", "Private network deployment"),
    ("GSMA: LTE to 5G Migration", "LTE", "related_to", "Migration strategies"),
    ("GSMA: LTE to 5G Migration", "5G NR", "related_to", "4G to 5G path"),
    ("Nokia: NTN Satellite", "NTN", "related_to", "Satellite 5G integration"),
    ("Nokia: 5G Security", "5G Core", "related_to", "Security architecture"),
    ("ITU: QoS in 5G", "TS 23.503", "references", "QoS framework"),
    ("ITU: IMT-2020 & 6G", "Release 18", "related_to", "5G-Advanced and beyond"),
    ("Samsung: V2X Sidelink", "V2X", "related_to", "V2X communications"),
    ("Samsung: V2X Sidelink", "Sidelink", "related_to", "Sidelink enhancements"),
    ("Huawei: 5GC SBA", "5G Core", "related_to", "Service-Based Architecture"),
    ("Huawei: 5GC SBA", "TS 23.502", "references", "5GS procedures"),
    ("Qualcomm: NR PHY", "5G NR", "related_to", "Physical layer design"),
    ("O-RAN Alliance: Open RAN", "O-RAN", "defines", "Open RAN architecture"),
    ("O-RAN Alliance: Open RAN", "TS 38.473", "references", "F1AP interface"),
]


def build_seed_graph():
    """Build initial graph with seed data."""
    G = nx.DiGraph()

    # Known whitepaper node names
    paper_names = {
        "Nokia: AI/ML in RAN", "Ericsson: 5G NR Architecture", "Ericsson: IMS in 5G",
        "Ericsson: Edge Computing", "Ericsson: Mission Critical",
        "GSMA: Network Slicing", "GSMA: Private 5G", "GSMA: LTE to 5G Migration",
        "Nokia: NTN Satellite", "Nokia: 5G Security",
        "ITU: QoS in 5G", "ITU: IMT-2020 & 6G",
        "Samsung: V2X Sidelink", "Huawei: 5GC SBA", "Qualcomm: NR PHY",
        "O-RAN Alliance: Open RAN",
    }

    for src, tgt, rel, desc in SEED_EDGES:
        for node in [src, tgt]:
            if node not in G.nodes:
                if node in paper_names:
                    ntype = "paper"
                elif node.startswith(("TS ", "TR ")):
                    ntype = "spec"
                elif node.startswith("Release"):
                    ntype = "release"
                else:
                    ntype = "concept"
                G.add_node(node, type=ntype, description="", source="seed_data")

        G.add_edge(src, tgt, relation=rel, description=desc, confidence=1.0)

    return G


def update_graph(G, entities, relations):
    """Update graph with new entities and relations."""
    added_nodes = 0
    added_edges = 0

    for entity in entities:
        if entity["name"] not in G.nodes:
            G.add_node(
                entity["name"],
                type=entity["type"],
                description="",
                source="extracted",
            )
            added_nodes += 1

    for rel in relations:
        if not G.has_edge(rel["source"], rel["target"]):
            G.add_edge(
                rel["source"],
                rel["target"],
                relation=rel["relation"],
                description=rel.get("context", ""),
                confidence=rel.get("confidence", 0.5),
            )
            added_edges += 1

    return added_nodes, added_edges


def render_graph(G, highlight_nodes=None, filter_type=None):
    """Render NetworkX graph as interactive Pyvis HTML."""
    # Apply filter
    if filter_type and filter_type != "all":
        nodes_to_show = [n for n, d in G.nodes(data=True) if d.get("type") == filter_type]
        # Also include neighbors of filtered nodes
        extended = set(nodes_to_show)
        for n in nodes_to_show:
            extended.update(G.predecessors(n))
            extended.update(G.successors(n))
        subgraph = G.subgraph(extended)
    else:
        subgraph = G

    net = Network(
        height="600px",
        width="100%",
        directed=True,
        bgcolor="#1a1a2e",
        font_color="white",
    )

    # Add nodes
    for node, data in subgraph.nodes(data=True):
        ntype = data.get("type", "concept")
        color = NODE_COLORS.get(ntype, "#f4a261")
        size = NODE_SIZES.get(ntype, 18)
        title = f"<b>{node}</b><br>Type: {ntype}<br>Source: {data.get('source', 'N/A')}"

        if highlight_nodes and node in highlight_nodes:
            color = NODE_COLORS["active"]
            size = 30

        net.add_node(
            node,
            label=node,
            color=color,
            size=size,
            title=title,
            font={"size": 12, "color": "white"},
        )

    # Add edges
    for src, tgt, data in subgraph.edges(data=True):
        rel = data.get("relation", "related_to")
        desc = data.get("description", "")
        title = f"{rel}: {desc}"
        net.add_edge(src, tgt, title=title, label=rel, font={"size": 8, "color": "#aaa"})

    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 150,
          "springConstant": 0.04
        },
        "stabilization": {"iterations": 100}
      },
      "interaction": {
        "hover": true,
        "navigationButtons": true,
        "keyboard": true,
        "tooltipDelay": 100
      },
      "edges": {
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
        "smooth": {"type": "dynamic"},
        "color": {"color": "#555", "highlight": "#e63946"}
      },
      "nodes": {
        "borderWidth": 2,
        "borderWidthSelected": 4,
        "font": {"size": 12}
      }
    }
    """)

    net.save_graph(GRAPH_OUTPUT_PATH)
    return GRAPH_OUTPUT_PATH


def save_graph_json(G):
    """Save graph in node-link JSON format."""
    from networkx.readwrite import json_graph
    data = json_graph.node_link_data(G)
    with open(GRAPH_JSON_PATH, "w") as f:
        json.dump(data, f, indent=2)
    return GRAPH_JSON_PATH


def load_graph_json():
    """Load graph from JSON file."""
    from networkx.readwrite import json_graph
    try:
        with open(GRAPH_JSON_PATH, "r") as f:
            data = json.load(f)
        return json_graph.node_link_graph(data, directed=True)
    except FileNotFoundError:
        return build_seed_graph()


def get_graph_stats(G):
    """Get graph statistics."""
    type_counts = {}
    for _, data in G.nodes(data=True):
        ntype = data.get("type", "unknown")
        type_counts[ntype] = type_counts.get(ntype, 0) + 1

    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "specs": type_counts.get("spec", 0),
        "releases": type_counts.get("release", 0),
        "concepts": type_counts.get("concept", 0),
        "papers": type_counts.get("paper", 0),
    }
