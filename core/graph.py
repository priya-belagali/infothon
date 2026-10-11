"""Dependency graph: port -> route -> shipment -> order  (NetworkX)."""
import networkx as nx


def build_graph(net):
    G = nx.DiGraph()
    for _, p in net["ports"].iterrows():
        G.add_node(f"port:{p.port_id}")
    for _, r in net["routes"].iterrows():
        G.add_edge(f"port:{r.port_id}", f"route:{r.route_id}")
    for _, s in net["shipments"].iterrows():
        G.add_edge(f"route:{s.route_id}", f"shipment:{s.shipment_id}")
    for _, o in net["orders"].iterrows():
        G.add_edge(f"shipment:{o.shipment_id}", f"order:{o.order_id}")
    return G


def exposed_orders(G, port_id):
    """Every order downstream of a closed port."""
    node = f"port:{port_id}"
    if node not in G:
        return []
    return sorted(n.split(":", 1)[1] for n in nx.descendants(G, node) if n.startswith("order:"))


def trace_path(G, port_id, order_id):
    return nx.shortest_path(G, f"port:{port_id}", f"order:{order_id}")
