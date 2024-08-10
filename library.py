"""library with classes and helper functions"""

import serial
import numpy as np

class Node():
    """ A class to represent an IoT node. """

    def __init__(self,name,port,threshold):
        """
        Constructs all necessary attributed of the Node object.

        Parameters:
        -----------
            name: str
                Name of the node.
            port: str
                Port number of the node.
            threshold: int
                Threshold cutoff of the .data section.
        Returns:
        --------
        None
        """
        self.name = name
        self.port = port
        self.threshold = threshold
        self.memory = []
        self.serial = serial.Serial(port,115200)
        self.serial.close()
    
    def connect(self):
        """
        Opens a connection to the device.
        
        Returns:
        --------
            None
        """
        self.serial.open()
        # print(f'Connected to device at: {self.port}')

    def disconnect(self):
        """
        Closes the connection to the device.
        
        Returns:
        --------
            None
        """
        self.serial.close()
        # print(f'Disconnected from device at: {self.port}')

class Network():
    """ A class to represent an IoT network. """
    def __init__(self,name,node_names,node_ports,node_thresholds):
        """
        Constructs all necessary attributed of the Node object.

        Parameters
        ----------
            name: str
                Name of the network.
            num_nodes: int
                Number of nodes.
            node_names: list of str
                List of node names.
            node_ports: list of str
                List of node ports.
            node_thresholds: list of int
                List of node .data thresholds.
        Returns
        -------
        None
        """
        self.name = name
        num_nodes = len(node_names)
        print(f'Initializing the "{self.name}" network.')
        try:
            self.nodes = [Node(node_names[i],node_ports[i],node_thresholds[i]) for i in range(num_nodes)]
            for i in self.nodes:
                print(f'-- {i.name}: success.')
        except Exception as e:
            print('-- Failed: Check the network configuration, replug the network, and try again. ', e)

    def memory_to_array(self):
        """
        Converts the memory lists of each node in the network to an array.
        
        Returns:
        --------
            None
        """
        for i in self.nodes:
            i.memory = np.array(i.memory)

    def get_golden_means(self):
        """
        Creates the golden reference for the nodes in the network.
        
        Returns:
        --------
            None
        """
        self.golden_means = np.array([i.memory.mean(axis=0) for i in self.nodes])

    def start(self):
        """
        Starts the serial connection to all devices in the network.
        
        Returns:
        --------
            None
        """
        for i in self.nodes:
            i.connect()

    def stop(self):
        """
        Stops the serial connection to all devices in the network.
        
        Returns:
        --------
            None
        """
        for i in self.nodes:
            i.disconnect()
