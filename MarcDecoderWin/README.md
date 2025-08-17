# Decoding repetition code with signaling LRU

The objective of this project is to see if in the "best" case (i.e. only leakage in CZs
and incoming errors with same probability) a decoder can use the information of the signaling
LRU to improve its logical performance.
We are going to explore this with a NN decoder because it has a lot of learning
capabilities (see https://arxiv.org/abs/2403.00706). 

The key question we want to answer:
can the NN decoder create a bigger performance gap between non-signaling LRUs and
signaling LRUs?

The data used to train and evaluate the NN decoder can be found in:

~/marc/data_external/20250402-signaling_lru_decoding_dicarlo_lab

it also includes a description of the data and simulation parameters.


## Initial thoughts

1. In https://arxiv.org/pdf/2403.00706, the logical error rate goes from 5.0% to
4.9% when including leakage information. The experiments did not include LRU and
the signaling of leakage was done with qutrit measurement. Average defect rates
of around 20%, while in the simulation data of this project they are around 10%.

2. Does the signaling LRU provide any extra information that the NN can use?
One can check by hand using stochastic leakage errors that some specific Pauli errors
caused due to leakage can be flagged by the singaling LRU and thus one can use
this extra information to better decode the syndromes. The problem is that the 
leakage errors are usually not very dominant thus this effect could be too small. 
**This should be checked.**

3. The effect of the LRU on the logical performance is 
increased for larger distance codes. 
In https://arxiv.org/pdf/2102.06131, they perform a repetition code with ancilla
resets (which can be seen as LRUs for the ancilla qubits) and see that the 
performance difference between using the LRU or not is only seen from d=13 to higher. 
They report that this is because the LRU is not ideal and the tradeoff of using the
LRU or not only is clear at high distances, meaning that the leakage effect is
more pronounced at higher distances. 
I believe google reported that the use of LRU did not impact the d=3 surface 
code but it reduced the logical error rate in the d=5 (check in which paper 
they report this).
**Here we are using a d=3 repetition code, which is the smallest one!**

4. Only ancillas leak and incoming data qubit errors are coherent.

5. The leakage and bit-flip probabilities are quite low which means that the logical
error probability at 20 rounds (maximum number of rounds for training) is ~10%
when decoded with a LUT. This means that very likely the logical error probability
for the NN will be smaller, which could make it difficult to check if there
is a difference between the signaling and non-signaling LRU. If we want to keep the
probabilities low, then we should go to higher number of rounds so that the 
logical error probability is at least 30% for the NN decoder.
**This needs to be seen.**

6. Signaling-LRU in the ancillas is like qutrit measurement followed by reset. 
Non-signaling-LRU in the ancillas is like qubit measurement followed by reset.
What's the difference between signaling-LRU and qutrit-meas+reset? Is it 
better performance or shorter total duration?

7. One can make a data-qubit signaling LRU "a-la-Google", i.e. transfer the leakage
population from the data qubits to the ancilla qubits and then reset the ancilla 
qubits. With the SLRU, we could know if the data qubits have been leaked, which
could help in decoding errors. 
**What is the reason tha Google does not do it? Is it because measuring+resetting 
takes more time than just resetting or because they cannot use the extra information
in graph-based decoders?**


## Addressing initial thoughts

2. Possible ways to check that:
- Plot fraction of 0s, 1s, and 2s. Compare the fraction of 1s and 2s.
    - The fraction of 2s is overall constant around ~4%, which is not huge. 
    However, the defect rates are around ~10% so they are comparable to ~4%...

3. Ask Leo if he can run the simulation for d=5 with the same noise model and
then check if there is a larger gap between SLRU and non-SLRU.


## NN decoding

Things to be though/done about:

1. Inputs to be given to the NN
2. Pre-processing of the data to have the same structure as we use for qrennd
3. Code the functions to load the data to the NN for the new inputs (and where to store them)
4. Architecture of the NN to use
5. SLRU vs NSLRU (convert the 2s to 1s), where to do that?
