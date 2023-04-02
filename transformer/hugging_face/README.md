# HuggingFace Tutorial for Transformer


## Basics

-   Transformer architecture - majorly focus on translation tasks
-   Three categories of Transformer models:
    -   **GPT**-like (also called **auto-regressive** Transformer models)
    -   **BERT**-like (also called **auto-encoding** Transformer models)
    -   **BART/T5**-like (also called **sequence-to-sequence** Transformer models)
-   History
    -   **June 2018**: <u>GPT</u>, the first pretrained Transformer model, used for fine-tuning on various NLP tasks and obtained state-of-the-art results
    -   **October 2018**: <u>BERT</u>, another large pretrained model, this one designed to produce better summaries of sentences (more on this in the next chapter!)-   
    -   **February 2019**: <u>GPT-2</u>, an improved (and bigger) version of GPT that was not immediately publicly released due to ethical concerns-  
    -   **October 2019**: <u>DistilBERT</u>, a distilled version of BERT that is 60% faster, 40% lighter in memory, and still retains 97% of BERT’s performance-  
    -   **October 2019**: <u>BART</u> and <u>T5</u>, two large pretrained models using the same architecture as the original Transformer model (the first to do so)- 
    -   **May 2020**: <u>GPT-3</u>, an even bigger version of GPT-2 that is able to perform well on a variety of tasks without the need for fine-tuning (called zero-shot learning)


## Transformers are Language models
-   GPT, BERT, BART, T5, etc.. have been trained as **language models**
-   Trained on large amounts of raw text in a self-supervised fashion
-   **Self-supervised learning** training - objective is automatically computed from the inputs of the model 
    -   Means humans are not needed to label the data
-   Develops a <u>statistical understanding</u> of the language it has been trained on
    -   Not very useful for specific practical tasks
    -   Because of this, general pretrained model goes through a process called **transfer learning**
    -   **transfer learning** - Model is fine-tuned in a supervised way — that is, using human-annotated labels — on a given task.
    -   Ex:  Predicting the next word in a sentence having read the n previous words
    -   Called **causal language modeling** because the output depends on the past and present inputs, but not the future ones.
    -    **masked language modeling**, in which the model predicts a masked word in the sentence.


## Transformers are big models
-   Apart from few outliers (like DistilBERT) - general strategy to achieve better performance is by <u>increasing the models’ sizes</u> as well as the <u>amount of data they are pretrained on</u>.
-   Training a model - Requires more data -> Costly (time and compute resources)
-   This is why sharing language models is paramount
    -   Sharing trained weights & building on top of already trained weights reduces overall compute cost & carbon footprint of the community.


## Transfer Learning
-   **Pretraining**
    -   Training model from scratch on large amounts of data (starts with randomly initialized weights - training without any prior knowledge)
    -   Requires large corpus and training can take up to several weeks
-   **Finetuning**
    -   Training done after model is pretrained
    -   To perform fine-tuning, 
        -   Acquire a pretrained language model -> perform additional training with a dataset specific to your task
    -   Why not simply train directly for the final task? Reasons:
        -   Pretrained model was already trained on a dataset that has some similarities with fine-tuning dataset
        -   Fine-tuning take advantages of knowledge acquired by the initial model during pretraining
        -   Ex: NLP problems, **pretrained model will have some kind of statistical understanding of the language you are using for your task**
        -   Since the pretrained model was already trained on lots of data, the **fine-tuning requires way less data to get decent results** - Thus, time and resources needed is also lesser.
    -   Ex: Leveraging a pretrained model trained on English language and then fine-tuning it on an arXiv corpus, resulting in a science/research-based model. The fine-tuning will only require a limited amount of data: the **knowledge the pretrained model has acquired is “transferred”**, hence the term transfer learning.
    -   Lower time, data, financial, and environmental costs. 
    -   Quicker & easier to iterate over different fine-tuning schemes (training is less constraining than a full pretraining)
    -   Achieves better results than training from scratch (unless you have lots of data)
    -   **Always try to leverage a pretrained model** — <u>one as close as possible to the task</u> needed — and fine-tune it


## Transformer architecture
-   Primarily consists of 2 blocks:
    -   **Encoder** - Receives input and builds a representation of the features of it (Model acquires understanding of I/P)
    -   **Decoder** - Uses encoder's representation (features) along with other inputs to generate target sequence (Model optimized to generate O/P)
-   Three types:
    -   **Encoder only models** - Good for tasks that require understanding of the input. Ex: Sentence classification, NER
    -   **Decoder only models** - Good for generative tasks such as text generation
    -   **Encoder-Decoder models** (or) **Seq-2-Seq models** - Good for generative tasks that require an input. Ex: Translation, Summarization
  

## Attention Layers
-   Transformers have attention layers 
-   **Pays specific attention to certain words in the sentence** you passed it (and more or less ignore the others) when dealing with the representation of each word
-   Ex: Task of translating text from English to French. 
    -   I/P: `“You like this course”`, 
    -   A translation model will need to also attend to the adjacent word `“You”` to get the proper translation for the word `“like”`, because in French the verb `“like”` is conjugated differently depending on the subject. Rest of the sentence, is not useful for the translation of that word. 
    -   In the same vein, when translating `“this”` the model will also need to pay attention to the word `“course”`, because `“this”` translates differently depending on whether the associated noun is masculine or feminine. Again, the other words in the sentence will not matter for the translation of `“this”`. 
-   Same concept applies to any task associated with natural language: 
    -   NEED FOR ATTENTION: A **word by itself has a meaning**, but that **meaning is deeply affected by the context**, which can be **any other word (or words) before or after the word being studied.**


## Original architecture
-   During training, encoder receives inputs (sentences) in a certain language and decoder receives the same sentences in the desired target language
-   In encoder, attention layers can use all words in a sentence (translation of a given word can be context dependent on what is after as well as before it in a sentence)
-   The decoder, however, works sequentially and can **only pay attention to the words in the sentence that it has already translated** (so, only the words before the word currently being generated)
-   For example, when we have predicted the first three words of the translated target, we give them to the decoder which then uses all inputs of encoder to try to predict fourth word
-   To speed things up during training,
    -   When the model has access to target sentences, the decoder is fed the whole target, but it is not allowed to use future words 
    -   Examples
        -   If it had access to the word at position 2 when trying to predict the word at position 2, it's not a problem
        -   When trying to predict the fourth word, the attention layer will only have access to the words in positions 1 to 3
-   Attention layers in Decoder
    -   **First attention layer** in a decoder block pays attention to all (past) inputs to the decoder
    -   **Second attention layer** uses the output of the encoder
    -   Thus, **decoder can access the whole input sentence to best predict the current word**
    -   Very useful as different languages can have grammatical rules that put the words in **different orders, or some context provided later in the sentence may be helpful** to determine the best translation of a given word
    -   **Attention mask** can also be used in encoder/decoder to prevent model from paying attention to some special words 
        -   — Ex: the special padding word used to make all the inputs the same length when batching together sentences


## Architectures VS Checkpoints
-   **Architecture**: Skeleton of the model — definition of each layer and each operation that happens within the model
-   **Checkpoints**: These are the weights that will be loaded in a given architecture
-   **Model**: This is an umbrella term that isn’t as precise as “architecture” or “checkpoint”: it can mean both
-   Ex: BERT is an architecture while bert-base-cased, a set of weights trained by the Google team for the first release of BERT, is a checkpoint. However, one can say “the BERT model” and “the bert-base-cased model”.


## Encoder models
-   Input sentence -> Encoder -> Feature vectors (numerical representation of the words)
-   Dimension of vector defined by architecture of model
    -   For base BERT model, it is 768
-   Good at extracting vectors carrying meaningful info about a sequence
-   **“bi-directional”** attention (context from left & right), **self-attention** - **auto-encoding** models
-   Representations of the word contains the value of the word, but **contextualised**
    -   Ex: I/P: `"Welcome to NYC"` -> Encoder -> Representation of `"to"` isn't representation of just `"to"`, also takes into account the words around it (context)
    -   Attention mechanism relates to different positions in a single sequence to compute the representation
-   Vector of 768 values holds meaning of the word within the text 
-   At each stage, attention layers can access all words in initial sentence 
-   Usecases
    -   Natural Language Understanding
    -   Seq classification, QA tasks and masked LM 
-   Examples
    -   BERT - Standalone encoder model at the time of release 
    -   DistilBERT
    -   ELECTRA
    -   RoBERTa
    -   ALBERT
-   Masked LM - task of predicting a hidden word in a sequence of words
    -   Ex: `"My ??? is Sylvian."` (BERT can predict it as `"name"` from the context taken from `"Sylvian"` on right)
    -   Neverthless, encoder model need to have a good understanding of the sequence, relationship/interdependence between words to predict it
-   Seq classification
    -   Sentiment Analysis
        -   Examples: 
            -   `"Even though I'm sad to see them go, I couldn't be more grateful"` - Positive
            -   `"I'm sad to see them go, I can't be grateful"` - Negative
        -   Both sentences use almost same words, but meaning is different and model is able to grasp the diff


## Decoder models
-   Example of popular decoder-only architecture is GPT-2
-   Input sentence -> Decoder -> Feature vectors (numerical representation of the words)
-   Ex: I/P: `"Welcome to NYC"` -> Decoder -> Representation of `"to"` isn't representation of just `"to"`, also takes into account the words around it (context)
-   Dimension of vector defined by architecture of model
-   **“uni-directional”** attention, **masked self-attention** - **auto-regressive** models
-   **Auto-regressive** - reuse their past O/P as I/P in the following steps
-  **Difference from encoders**: 
   -  Masked self-attention, words can only see words on left side, right side is hidden. Ex: `"to"` vector is unmodified by `"NYC"` word. Right context of the word is masked.
   -  **Decoders only have access to a single context** (left or right context)
-   Pretraining of decoder models usually revolves around predicting the next word in the sentence
-   Usecases
    -   **Text Generation**
-   Examples
    -   CTRL
    -   GPT
    -   GPT-2
    -   Transformer XL
-   **Causal Language Modelling** (Guessing next word in a sentence)
    -   Ability to generate a word/sequence of words given a sequence of words
-   GPT-2 has maximum context size of 1024, we could generate upto 1024 words
  