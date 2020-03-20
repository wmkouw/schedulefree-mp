# Schedule-free variational message-passing

Message passing in factor graphs should be free of an explicit message scheduler. The scheduler is the bottleneck in computational cost, because it has to parse the graph in its entirety. By passing messages based on a local, distributed criterion, the scheduler can be circumvented. Schedule-free MP scales better to to larger graphs.

### Comments
Feedback can be submitted to the [issues tracker](https://github.com/wmkouw/schedulefreeMP/issues).
