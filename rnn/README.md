## References:

- https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
- https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21

## Notes:


### RNN Basics
- Words converted to machine readable vectors
- RNN processes the sequence of vectors one by one
- Passes previous hidden state to next step of the sequence
- RNN suffers from **short-term memory**
    - Long sequence makes it hard to carry info from earlier time steps to later ones
- During backprop, RNN suffer from **vanishing gradient problem** 
    - It is when the **gradient shrinks** as it back propagates through time
    - If a gradient value (dw) becomes extremely small, it doesn’t contribute too much learning (w = w - alpha * dw)
- **LSTM**’s and **GRU**’s were created as the solution to short-term memory
    - Uses internal mechanisms called **gates** that can regulate the flow of information
    - Gates can learn which data in a sequence is important to keep or throw away
        - Passes relevant information down the long chain of sequences to make predictions


### RNN Summarised & Usecases
- Vanilla RNN - Good for processing sequence data for predictions but suffer from short-term memory
- LSTMs and GRUs - methods to mitigate short-term memory using gates mechanism
- Gates - Neural networks to regulate flow of info being passed from one time step to next
- Usecase: Speech Recognition, Speech Synthesis, Natural Language Understanding, etc..


### Hidden State
- Hidden state (memory) - holds info of previous data that neural network has seen before
```
Xt = input
Ht = new hidden state
Ct = new cell state
Ht-1 = previous hidden state
Ct-1 = previous cell state
```
- How is it calculated?
    - Input in previous hidden state `(Ht-1)` is combined with current input `(Xt)` to form a vector
    - `(Ht-1 + Xt)` -> Making it have the information of the current as well as previous inputs
    - `Ht = tanh(Ht-1 + Xt)` 
    - **Tan activation** done to **regulate values** flowing through network
        - Output values between **-1 and 1**
        - **If not regulated**, when undergoing many transformations in the neural network due to various math operations, **some values can explode**
        - Whereas, tan ensures values stay between -1 and 1, regulating neural network's output


### LSTM
- **Cell state** (memory of the network) - carries relative info throughout sequence processing - Reducing effects of short-term memory
- Info gets added/removed to the cell via gates
- **Gates** 
    - Different neural network which decided which info is allowed on cell state
    - **Learns relevant info to keep or forget during training**
    - Contains **sigmoid activation** - Squishes values between **0 and 1**
        - Helpful to update/forget data (`X*0=0` causing values to disappear & `X*1=X` causing values to stay)
- 3 different gates
    - **forget gate**
    - **input gate**
    - **output gate**
- **forget gate**
    - `sigmoid(Ht-1 + Xt)` -> `closer to 0 = forget` and `closer to 1 = keep`
- **input gate** 
    - tanh activation to squish values between -1 and 1 and sigmoid activation done to know which is important and which is not from the tan output, both are then multiplied
    - Input gate output = `sigmoid(Ht-1 + Xt)` multiplied with `tanh(Ht-1 + Xt)`
- **cell state**
    - Forget vector multiplied with previous cell state (possibility of dropping values in cell state)
    - Pointwise addition of `Ct-1` with output from the input gate which updates the cell state to new values `Ct`
- **output gate**
    - Output is the hidden state
    - Output gate output = `sigmoid(Ht-1 + Xt)` multiplied with `tanh(Ct)` 
    - Sigmoid determines what info to carry in hidden state
- New cell state and new hidden state (output gate output) are then passed to next time step


### GRU
- Similar to LSTM, gets rid of cell state & uses hidden state to transfer info
- 2 different gates
    - **reset gate**
    - **update gate**
- **update gate**
    - Similar to forget gate in LSTM
- **reset gate**
    - Decides **how much past info to forget**
- Lesser tensor operations, speedier to train