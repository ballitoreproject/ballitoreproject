from .imports import *
import networkx as nx
import matplotlib.pyplot as plt
import ipycytoscape
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def get_family_tree_network():
    df_edges = pd.read_csv(os.path.join(PATH_DATA, "familytree.csv"))
    df_nodes = pd.read_csv(os.path.join(PATH_DATA, "people.csv"))

    G = nx.DiGraph()
    for i, row in df_edges.iterrows():
        G.add_edge(row.person1, row.person2, relationship=row.relationship)

    for row in df_nodes.to_dict("records"):
        node = row.pop("person")
        if G.has_node(node):
            for k, v in row.items():
                G.nodes[node][k] = v
            
    for node in G.nodes():
        G.nodes[node]["is_family"] = True

    return G


def get_node_positions(G):
    import pygraphviz as pgv

    G_viz = pgv.AGraph(strict=False, directed=True)
    G_viz.graph_attr["rankdir"] = "TB"

    for n, d in G.nodes(data=True):
        G_viz.add_node(n, **d)
    for a, b, d in G.edges(data=True):
        G_viz.add_edge(a, b, **d)

    # Use Graphviz to calculate positions
    G_viz.layout(prog="dot")
    node2pos = {}
    for node in G_viz.nodes():
        x, y = node.attr["pos"].split(",")
        node2pos[node.name] = (float(x), float(y))
    return node2pos


def get_correspondence_network(df=None, incl_family_tree=True, min_num_letters=10):
    from .ballitoreproject import get_data

    G = nx.DiGraph() if not incl_family_tree else get_family_tree_network()
    dfx = (
        get_ballitore_data()
        .query('sender!="" & recipient!=""')
        .groupby(["sender", "recipient"])[["num_letters", "num_words"]]
        .sum()
        .reset_index()
        .query(f"num_letters>={min_num_letters}")
        if df is None
        else dfx
    )
    for i, row in dfx.iterrows():
        if G.has_edge(row.sender, row.recipient):
            G.edges[(row.sender, row.recipient)]["num_letters"] = row.num_letters
            G.edges[(row.sender, row.recipient)]["num_words"] = row.num_words
        # elif G.has_node(row.sender) or G.has_node(row.recipient):
        else:
            G.add_edge(
                row.sender,
                row.recipient,
                relationship="wrote",
                num_letters=row.num_letters,
                num_words=row.num_words,
            )
    
    for n,d in G.nodes(data=True):
        if not 'is_family' in d:
            G.nodes[n]['is_family']=False

    return G




import ipycytoscape
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_network(G, edge_width='weight', 
                 node_color=None, edge_color=None,
                 min_width=1, max_width=10,
                 node_colormap='cividis', edge_colormap='plasma'):
    cyto = ipycytoscape.CytoscapeWidget()
    
    def get_color_mapping(values, colormap):
        if not values:
            return {}
        unique_values = list(set(values))
        if all(isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.', '', 1).isdigit()) for v in unique_values if v != '?'):
            numeric_values = [float(v) for v in unique_values if v != '?']
            min_val, max_val = min(numeric_values), max(numeric_values)
            norm = plt.Normalize(min_val, max_val)
            cmap = plt.get_cmap(colormap)
            return {str(v): mcolors.rgb2hex(cmap(norm(float(v)))) for v in unique_values if v != '?'}
        else:
            cmap = plt.get_cmap(colormap, len(unique_values))
            return {str(val): mcolors.rgb2hex(cmap(i)) for i, val in enumerate(unique_values)}
    
    # Prepare style
    style = [
        {
            'selector': 'node',
            'style': {
                'content': 'data(label)',
                'text-valign': 'center',
                'text-halign': 'center',
                'text-outline-width': 1,
                'text-outline-color': '#888',
                'color': '#fff',
                'font-size': '12px',
                'text-wrap': 'wrap',
                'text-max-width': '200px'
            }
        },
        {
            'selector': 'edge',
            'style': {
                'content': 'data(label)',
                'curve-style': 'bezier',
                'target-arrow-shape': 'triangle',
                'font-size': '10px',
                'text-rotation': 'autorotate',
                'text-wrap': 'wrap',
                'text-max-width': '200px'
            }
        },
        {
            'selector': 'node:selected',
            'style': {
                'border-width': '3px',
                'border-color': '#DAA520'
            }
        },
        {
            'selector': 'edge:selected',
            'style': {
                'width': '5px',
                'line-color': '#DAA520',
                'target-arrow-color': '#DAA520'
            }
        }
    ]
    
    # Node color styling
    if node_color:
        node_values = [str(G.nodes[n].get(node_color, '?')) for n in G.nodes()]
        node_colors = get_color_mapping(node_values, node_colormap)
        for value, color in node_colors.items():
            style.append({
                'selector': f'node[{node_color} = "{value}"]',
                'style': {
                    'background-color': color,
                }
            })
    
    # Edge color styling
    if edge_color:
        edge_values = [str(G.edges[e].get(edge_color, '?')) for e in G.edges()]
        edge_colors = get_color_mapping(edge_values, edge_colormap)
        for value, color in edge_colors.items():
            style.append({
                'selector': f'edge[{edge_color} = "{value}"]',
                'style': {
                    'line-color': color,
                    'target-arrow-color': color
                }
            })
    
    # Edge width styling
    if edge_width:
        edge_values = [float(G.edges[e].get(edge_width, 0)) for e in G.edges()]
        if edge_values:
            min_val, max_val = min(edge_values), max(edge_values)
            for e in G.edges():
                value = float(G.edges[e].get(edge_width, 0))
                if min_val != max_val:
                    width = ((value - min_val) / (max_val - min_val)) * (max_width - min_width) + min_width
                else:
                    width = (min_width + max_width) / 2
                style.append({
                    'selector': f'edge[id = "{e[0]}-{e[1]}"]',
                    'style': {
                        'width': str(width)
                    }
                })
    
    # Prepare graph data with tooltips
    graph_data = {
        "nodes": [
            {
                "data": {
                    "id": str(n),
                    "label": str(n),
                    node_color: str(G.nodes[n].get(node_color, '?')) if node_color else '?',
                    "tooltip": "<br>".join([f"{k}: {v}" for k, v in G.nodes[n].items()])
                }
            } for n in G.nodes()
        ],
        "edges": [
            {
                "data": {
                    "source": str(u),
                    "target": str(v),
                    "id": f"{u}-{v}",
                    "label": G.edges[u, v].get('relationship', ''),
                    edge_color: str(G.edges[u, v].get(edge_color, '?')) if edge_color else '?',
                    "tooltip": "<br>".join([f"{k}: {v}" for k, v in G.edges[u, v].items()])
                }
            } for u, v in G.edges()
        ]
    }
    
    cyto.graph.add_graph_from_json(graph_data, directed=True)
    cyto.set_style(style)
    
    # Set up tooltip
    cyto.set_tooltip_source('tooltip')
    
    return cyto
