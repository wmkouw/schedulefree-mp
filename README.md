# Reactive Message-Passing

We want message passing in ForneyLab to be _reactive_: each node should only respond to incoming messages if sending out a message will reduce free energy sufficiently. In order to make this decision, the nodes and edges need access to local free energies.

Here I experiment with how reactive message passing should occur in various factor graphs.

The repo is closed until deemed mature.
