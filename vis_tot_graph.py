import pickle
#from ToT_Exp import MyChecker
# Load the object from a file

class IgnoreMissingClassUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError):
            # Provide a default implementation or a dummy class
            return type('MissingClass', (object,), {})

# Load the object using the custom unpickler
with open('classtree.pkl', 'rb') as file:
    loaded_object = IgnoreMissingClassUnpickler(file).load()

# Use the loaded object
print(loaded_object)  # This will print an instance of MissingClass if the original class is not found
tot_chain = loaded_object
breakpoint()
import pygraphviz as pgv
def add_nodes_edges(graph, level):
    if len(tot_chain.tot_memory.stack[level].children) == 0:
        return
    node_data = tot_chain.tot_memory.stack[level]
    label = node_data.text
    graph.add_node(level, label=label, shape='box', style='filled', fillcolor='lightyellow', color='black')
    children = node_data.children
    for child, child_data in children.items():
        add_nodes_edges(graph, child)
        graph.add_edge(level, child)

# Create a new graph
G = pgv.AGraph(directed=True)

# Add the root node with a textbox style
root = 0  # Assuming the root is at level 0
root_data = tot_chain.tot_memory.stack[root]
root_label = root_data.text
G.add_node(root, label=root_label, shape='box', style='filled', fillcolor='lightyellow', color='black')
add_nodes_edges(G, root)

# Layout and render the graph
G.layout(prog='dot')
G.draw('tree.png')
#######