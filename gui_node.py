# Node in the GUI

class GuiNode():

    def __init__(self, C):
        """ Constructor

            ### Arguments:
                self<GuiNode>: self-reference
                C<list<float>>: coordinates
        """
        self.X = C[0]
        self.Y = C[1]

        self.assignment = None
        self.connected_nodes = []

    def assign(self, value):
        """ Assigns the a label to the node

            ### Arguments:
                self<GuiNode>: self-reference
                value<int>: label to assign
        """
        self.assignment = value

    def connect(self, node, backtrace=True):
        """ Connects two nodes

            ### Arguments:
                self<GuiNode>: self-refrence
                node<GuiNode>: node to connect
                backtrace<boolean>: if True, has other node connect
        """
        if not node in self.connected_nodes:
            self.connected_nodes.append(node)
        if backtrace:
            node.connect(self, backtrace=False)

    def is_assignable(self):
        """ Checks if the node is assignable

            ### Arguments:
                self<GuiNode>: self-reference
            
            ### Returns:
                assignable: returns True if assignment is None and a neighbor is assigned, returns False if already assigned, returns None if assignment is None and all neighbords are None
        """
        if self.assignment is not None:
            return False
        for node in self.connected_nodes:
            if node.assignment is not None:
               return True
        return None

    def assign_by_neighbor(self):
        """ Assigns the node based on neighbors

            ### Arguments:
                self<GuiNode>: self-reference
        """
        labels = {}
        for node in self.connected_nodes:
            if node.is_assignable() == False:
                label = node.assignment
                if not label in labels:
                    labels[label] = 0
                labels[label] += 1
        best_label = None
        best_score = None
        for label in labels:
            if best_score is None or labels[label] > best_score:
                best_score = labels[label]
                best_label = label
        self.assign(best_label)

    def within_range(self, x, y, r=50):
        """ Checks if a node is within an r radius box of x,y

            ### Arguments:
                self<GuiNode>: self-reference
                x<float>: x start
                y<float>: y start
                r<float>: radius

            ### Returns:
                within<boolean>: if within, return True
        """
        if self.X > x-r and self.X < x+r and self.Y > y-r and self.Y < y+r:
            return True
        return False
                

def assign_nodes(nodes, logger, indent_level=6):
    """ Assigns all nodes by iteratively assigning unassigned nodes with assigned neighbors

        ### Arguments:
            nodes<dict<GuiNode>>: dictionary of gui Nodes
            logger<QuickLogger>: logger
            indent_level<int>: indentation level for logger
    """
    unassignable = len(nodes)
    while unassignable > 0:
        unassignable = 0
        assigned = 0
        assignable = 0
        to_assign = []
        for ii in nodes:
            assignability = nodes[ii].is_assignable()
            if assignability == None:
                unassignable += 1
            elif assignability == False:
                assigned += 1
            else:
                assignable += 1
                to_assign.append(nodes[ii])
        logger.progress(unassignable/len(nodes), "Assigning nodes. {} assigned, {} assignable, {} unassignable".format(assigned, assignable, unassignable), indent_level=indent_level, total_size=150)

        for node in to_assign:
            node.assign_by_neighbor()

    logger.progress(1, "Assigned {} nodes".format(len(nodes)), indent_level=indent_level, total_size=150, finish=True)
        



