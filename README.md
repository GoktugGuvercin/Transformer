
# Transformer

Transformer is a sequence transduction model that primarily employs an attention mechanism. It typically adopts an encoder-decoder architecture; nevertheless, it can also function effectively as encoder-only or decoder-only structures, such as BERT and GPT. 

Encoders receive raw input sequences and convert them into continuous vector representations. This transformation performs a seq-to-seq operation, but actually changes the representation space of the input. While doing that, it reveals and highlights contextual details by evaluating the correlations between the tokens. Decoders, on the other hand, reverts this operation; it aims to generate a new sequece of tokens, utilizing encoder-generated representations.

- Machine Translation
- Text Summarization
- Speech Recognition
- Dialogue Generation

In these applications, decoders can restructure input sequences, shorten it to capture a broader meaning, generate conversational responses or translate languages. The unifying factor among these tasks is the necessity of generating a new sequence, which exclusively relies on the decoder. 

If we do machine translation and our input given to the encoder is *"I read a book"*, we expect the decoder to generate *"Ich lese ein Buch"* as a predicted output. The first problematic issue is how the decoder learns German language system and the organization of the words in a typical German sentence. Without this, it is not possible to translate the given English sentence. That is why, ground-truth labels are given to the decoder as input; it observes the main structure of target language and the distribution of the words in a sentence organization scheme. 

Let's think about the analogy of teaching a child a language she has never encountered. If her first language is English and she has had no exposure to German or French, she cannot translate an English sentence into these languages. However, once we provide her with the answer, she begins to recognize how subjects, verbs, and objects relate to one another in target and source languages. In other words, even humans are unable to generate something that they have never been exposed to. This obliges ground-truth label to be fed into the decoder as input aside from encoder's output. 


## Decoder

A transformer decoder comprises a stack of $N$ number of identical blocks, each containing:

- Masked Multi-Head Attention
- Multi-Head Attention
- Layer Norm
- Feed Forward Neural Network


Before decoding, ground-truth sentence in German is tokenized and converted into continuous vector embeddings. Then, positional encodings are added to these ebmeddings to provide a sense of order, since self-attention system does not encode positional information inherently. Similarly, the input sequence in English undergoes an analogous process to prepare it for the encoder. 

While training the entire architecture, we expect the decoder to learn how to predict the next token for the all tokens in target sequence. In other words, the decoder needs to learn;

- Given \<start\>, predict "Ich"
- Given \<start\> Ich, predict "lese"
- Given \<start\> Ich lese, predict "ein"
- Given \<start\> Ich lese ein, predict "Buch"

At this point, the decoder is enforced to construct the whole sentence, but the prediction of one word only relies on its preceding words. In the training stage, the decoder will be exposed to so many different sentences; this allows it to learn which word should come before or after which word, and the coherence between subjects, verbs and objects. However, this learning takes time; it will not be so quickly. That is why, the predictions of the decoder will be mostly instable, and gramatically or semantically wrong. 

Predicting a token is conditioned on the previous tokens in the training stage. If the initial tokens are incorrectly predicted, and we provide the decoder with those predictions, the next tokens that will be predicted will be naturally affected and thereby being incorrect. That is why, decoder is not fed by its predicted tokens, instead always receive the correct tokens in target sentence. In that way, we would force it to stay on the correct track. This is called *"teacher forcing"*. A simple example is given below:

<div align="center">

| Step   | Decoder Input          | Prediction |
|--------|------------------------|------------|
| Step 1 | \<start\>              | Ich        |
| Step 2 | \<start\> Ich          | trinke     |
| Step 3 | \<start\> Ich lese     | kein       |
| Step 4 | \<start\> Ich lese ein | Buch       |
  
</div>


If predicted tokens were used to estimate the next token in training stage, it would proceed one at a time because we don't know what the model predict. On the other hand, teacher forcing enables us to train the entire architecture in parallel because we know what ground-truth sequence is. To understand this in a more clear way, let's look at masked multi-head attention.

The decoder has to learn how to predict the next token for every token in the sentence, and it is only allowed to consider the previous tokens. This means that if the decoder will predict $N$'th token, all the later tokens should be out of consideration. That is why masked multi-head attention defines an upper triangular masking matrix to hide the tokens coming after the token that will be predicted.

<p align="center">
  <img src="https://github.com/GoktugGuvercin/Transformer/blob/main/images/masked%20attention%20matrix.png" width="700" title="Masking tokens in parallel">
</p>

If we have five tokens in the sentence, the decoder has to learn how to predict each of these five tokens, but in parallel not sequentially. To parallelize them, the prediction of every token is considered in the matrix; it covers all scenarios. Additionally, each token can only attend to itself and earlier tokens. Hence, masken attention system enforce causality; how the next token will be predicted is reasoned to previous tokens. This underlies autoregressive nature of the decoder. 

In masked self attention, query and key, both of which refers to continuous representation of target sentence, are at first multiplied, and then the result of this multiplication is normalized by square root of feature dimension. This generates raw attention scores. Masking is applied at this point. 

<p align="center">
  <img src="https://github.com/GoktugGuvercin/Transformer/blob/main/images/masked%20attention%20matrix.png" width="700" title="Masked Multi Head Attention">
</p>