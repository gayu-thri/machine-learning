# seq2seq

### Word to word translation
- Consider the example, 
  - If English is translated to Koren
    - The input & output sentence pair are as below
        - English I/P: "i love you"
        - "i" -> "nan"
        - "love" -> "saranghey"
        - "you" -> "nul"
        - Korean O/P: "nan saranghey nul"
- ISSUE:
  - Why word to word translation is not an optimal solution? Issues below
    - order can't be mapped directly based on the same position of occurence
    - num_words_input_sentence may not be equal to num_words_output_sentence

### Sequence of words translation using RNN
- Encoder and Decoder RNNs will be used
- Alias encoder decoder architecture (or) seq2seq model
- Context vector representing the whole input sentence (het) is passed from encoder to decoder
- First timestep of decoder
  - hd0 = hidden state of decoder = context vector from encoder = het
  - id0 = input of decoder = `<start>` (sign for starting the translation)
  - od0 = output of decoder =  "nan" (first word in Korean O/P)
- Second timestep of decoder
  - Input in previous hidden state `(Ht-1)` is combined with current input `(Xt)` to form a vector
    - `(Ht-1 + Xt)` -> Making it have the information of the current as well as previous inputs
    - `Ht = tanh(Ht-1 + Xt)` 
    - **Tan activation** done to **regulate values** flowing through network
        - Output values between **-1 and 1**
        - **If not regulated**, when undergoing many transformations in the neural network due to various math operations, **some values can explode**
        - Whereas, tan ensures values stay between -1 and 1, regulating neural network's output
- Final timestep of decoder
  - output of decoder = `<end>` (sign of finished translation)
- ISSUE: 
  - Encoder encodes full source sentence into a fixed length context vector
  - Not all info can be stored in one single vector due to long-term dependency especially when input is a long sentence
- SOLUTION:
  - Use encoder's each state with the current state of decoder to generate dynamic context vector
    - i.e, encoder info in a sequence of vectors & not in single context vector
  - Choose only what's needed from encoder at that particular decoder's timestamp


### Attention
- First, the encoder passes a lot more data to the decoder. 
- Instead of passing the last hidden state of the encoding stage, the encoder passes all the hidden states to the decoder.
- In a Seq2Seq with attention model, 
  - Decoder hidden state in the first time step of the decoder is usually **initialized with a fixed value, such as a zero vector or a learned vector**. 
    - Because there is no previous decoder output or context vector to use as input for the first decoder time step.
- Steps in detail,
    1. The attention decoder RNN takes in the embedding of the <END> token, and an initial decoder hidden state.
    2. The RNN processes its inputs, producing an output and a new hidden state vector (h4). The output is discarded.
    3. Attention Step: We use the encoder hidden states and the h4 vector to calculate a context vector (C4) for this time step.
    4. We concatenate h4 and C4 into one vector.
    5. We pass this vector through a feedforward neural network (one trained jointly with the model).
    6. The output of the feedforward neural networks indicates the output word of this time step.
    7. Repeat for the next time steps
- Steps in layman terms,
    1. Compute the score for each encoder output at the current decoder time step using a score function. One common score function is the dot-product attention, where the score is the dot product of the decoder hidden state and the encoder output.
    2. Apply a softmax function to the scores to obtain the attention weights, which represent the importance of each encoder output in generating the current decoder output.
    3. Compute the context vector as the weighted sum of the encoder outputs, where the weights are the attention weights.
    4. Concatenate the context vector with the current decoder hidden state and feed it through a linear layer to generate the next decoder hidden state.
    5. Use the decoder hidden state to generate the output token.
    6. Repeat steps 1-5 for each time step in the decoder sequence.