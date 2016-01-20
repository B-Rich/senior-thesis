# Designing quantum algorithms for qubit control calibration

## Peter Karalekas
## Thesis Advisor: Robert Schoelkopf
## Mentor: Brian Vlastakis

### Abstract

One of the most important requirements for building a quantum computer is having complete control over qubits and quantum gates. That is, errors in qubit state preparation and measurement, as well as errors in the fidelity of single-qubit gates need to to be sufficiently low for quantum computation to be feasible. One of the most widely accepted methods of diagnosing errors in gate implementation is that of randomized benchmarking, a process in which a quantum circuit composed of randomly chosen, yet known, gates is applied to a qubit prepared in the ground state, and the final state is then measured. The randomness of the circuit allows for the extraction of an average error per gate independent of the individual gates themselves, effectively evaluating the gate implementation process as a whole. An additional method known as interleaved randomized benchmarking can be used in conjunction to separate out the errors due to individual gates. For my thesis, I wrote a pulse generation and simulation software for randomized benchmarking in the Python programming language, and was then able to run it on a physical quantum system composed of superconducting qubits. I hope this tool will prove useful as the lab works to improve pulse tuning methods and ultimately gate fidelities of their quantum operations.