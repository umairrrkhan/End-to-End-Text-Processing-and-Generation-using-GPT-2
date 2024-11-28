# End-to-End Text Processing and Generation using GPT-2

## Overview

EnglishTextGen is a fine-tuned GPT-2 model designed for generating high-quality English text. The model was trained on a subset of the Common Corpus dataset, which is part of the AI Alliance Open Trusted Data Initiative. This dataset contains over 2 trillion tokens of permissibly licensed content with provenance information. For this project, a cleaned subset of 39 million tokens was used to fine-tune the model.

## Dataset

The dataset used for training EnglishTextGen is a subset of the Common Corpus dataset, which is the largest fully open multilingual dataset for training large language models (LLMs). The dataset contains over 2 trillion tokens of permissibly licensed content with provenance information. For this project, a cleaned subset of 39  million tokens was used well we can used more but bcz of the compute power i cant but if u have a good gpu u can scale it up u just have to change the file and u can combine too u will find the better result , with approximately 100 % of the tokens being in English and covering a wide range of topics and styles.

## Training Process

The model was fine-tuned using the Hugging Face Transformers library. The training process involved the following steps:

1. **Data Cleaning**: The dataset was cleaned to remove low-quality and irrelevant data. The cleaning process included removing programming language entries, cleaning text content, detecting English language, and filtering only English content.
2. **Tokenization**: The text data was tokenized using the GPT-2 tokenizer, with padding and truncation applied to handle input sequences of varying lengths.
3. **Training**: The model was fine-tuned for 3 epochs with a batch size of 8 and a learning rate of 2e-5. The training process was monitored using the training loss and performance on a validation set.
4. **Evaluation**: The model was evaluated using metrics such as perplexity, BLEU score. Human evaluation was also conducted to gain insights into the model's strengths and weaknesses.

## Model Architecture

The model architecture is based on the GPT-2 model from the Hugging Face Transformers library. The model was fine-tuned on the cleaned subset of the Common Corpus dataset to improve its performance on English text generation tasks.

## Usage

To use EnglishTextGen for generating English text, follow these steps:

1. **Install Dependencies**:
   ```bash
   pip install transformers torch

2. **Load the Model and Tokenizer**:
   ```bash
   from transformers import GPT2Tokenizer, GPT2LMHeadModel
   model = GPT2LMHeadModel.from_pretrained('./nano_gpt_model')
   tokenizer = GPT2Tokenizer.from_pretrained('./nano_gpt_model')
   tokenizer.pad_token = tokenizer.eos_token

3. ***Generate Text**:
   ```bash
   def generate_text(prompt, max_length=100, num_return_sequences=1, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.2):
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text
   # Example usage
   prompt = "Once upon a time in a faraway land, there lived a brave knight. Generate a short story based on this prompt."
   generated_text = generate_text(prompt)
   print(generated_text)

4. ***Evaluation***:
- The model was evaluated using metrics such as perplexity, BLEU  score. Human evaluation was also conducted to gain insights into the model's strengths and weaknesses. The model demonstrated the ability to generate coherent and relevant English text based on the given prompts.

5. **Contributing**:
- Contributions are welcome! Please feel free to open an issue or submit a pull request if you have any suggestions or improvements.

6. **Acknowledgments**:
- Thanks to the AI Alliance Open Trusted Data Initiative for providing the Common Corpus dataset.
- Thanks to the Hugging Face team for providing the Transformers library.

7. **Dataset Link**:
 - [Common Corpus](https://huggingface.co/datasets/PleIAs/common_corpus)
