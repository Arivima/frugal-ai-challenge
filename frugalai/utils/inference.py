import re
from tqdm import tqdm
from datasets import Dataset
from data_format import create_formatting_function

def output_parser(output):
    """takes a decoded llm output as argument and returns input quote and predicted label"""
    def first_digit(sentence):
        match = re.search(r'\d', sentence)
        if match:
            return int(match.group())
        return -1
    
    # compatible with list of outputs or single output
    if isinstance(output, list) and output:
        text = output[0]
    else:
        text = output

    # get the quote and label parts from the output
    try:
        after_statement = text.split("### Statement to classify:", 1)[1]
        quote, answer_part = after_statement.split("### Answer:", 1)
        quote = quote.strip()
        answer_part = answer_part.strip()
        label = first_digit(answer_part)
        return quote, label
    except Exception as e:
        print(f"Parsing error: {e}")
        return text.strip(), -1, f"Error: {str(e)}"


def create_parser_function(
        statement_delimitator="### Statement to classify:", 
        answer_delimitator="### Answer:"
        ):
    """creates a custom output parser function based on specific delimitators"""

    def output_parser(output):
        """takes a decoded llm output as argument and returns input quote and predicted label"""
        def first_digit(sentence):
            match = re.search(r'\d', sentence)
            if match:
                return int(match.group())
            return -1
        
        # compatible with list of outputs or single output
        if isinstance(output, list) and output:
            text = output[0]
        else:
            text = output

        # get the quote and label parts from the output
        try:
            after_statement = text.split(statement_delimitator, 1)[1]
            quote, answer_part = after_statement.split(answer_delimitator, 1)
            quote = quote.strip()
            label = first_digit(answer_part)
            return quote, label
        except Exception as e:
            print(f"Parsing error: {str(e)}")
            return text.strip(), -1
        
    return output_parser
    

def predict(
    model,
    tokenizer,
    max_seq_length = 2048,
    statements = ["Climate change is just a hoax created to control people."], # default test
    labels_true = [1], # default test
    ):
    def tokenize(element):
        return tokenizer(
            element,
            max_length=max_seq_length,
            return_tensors = "pt"
            ).to("cuda")

    format_func = create_formatting_function(tokenizer) # get the formating function for that model
    inputs = format_func(statements) # applies the chat template / prompt template
    tokenized_texts = [tokenize(input) for input in inputs] # tokenize inputs

    decodeds = []
    quotes = []
    labels_pred = []
    for tokenized_text in tqdm(tokenized_texts):
        outputs = model.generate(
            **tokenized_text, 
            max_new_tokens = 64, 
            use_cache = True
            ) # generate tokens
        decoded = tokenizer.batch_decode(outputs) # decode
        quote, label = output_parser(decoded) # extracts the category digit

        decodeds.append(decoded)
        quotes.append(quote)
        labels_pred.append(label)

    return Dataset.from_dict({
        'decoded' : decodeds,
        'quotes': quotes,
        'label_true': labels_true,
        'label_pred': labels_pred,
        'result': [str(t) == str(p) for t, p in zip(labels_true, labels_pred)],
    })



# batch inference WIP
def batch_inference(
    model,
    tokenizer,
    format_func,
    max_seq_length,
    statements,
    labels_true,
    batch_size=8  # Adjust based on your GPU memory
    ):

    # Prepare all inputs
    inputs = format_func(statements)
    results = {
        'decoded': [],
        'quotes': [],
        'label_true': labels_true,
        'label_pred': [],
        'result': []
    }

    # Process in batches
    for i in tqdm(range(0, len(inputs), batch_size)):
        batch_inputs = inputs[i:i+batch_size]
        batch_labels = labels_true[i:i+batch_size]

        # Tokenize the batch
        tokenized_batch = tokenizer(
            batch_inputs,
            max_length=max_seq_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to("cuda")

        # Generate outputs
        outputs = model.generate(
            **tokenized_batch,
            max_new_tokens=64,
            use_cache=True
        )

        # Decode outputs
        decoded_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Parse results
        for j, decoded in enumerate(decoded_batch):
            quote, label = output_parser(decoded)

            results['decoded'].append(decoded)
            results['quotes'].append(quote)
            results['label_pred'].append(label if label is not None else -1)

    # Calculate results after all batches
    results['result'] = [
        str(t) == str(p) and p is not None
        for t, p in zip(results['label_true'], results['label_pred'])
    ]

    return Dataset.from_dict(results)




# @title output_parser_func - not needed with this way - we generate logits
import re

def create_parser_function(
        statement_delimitator="### Statement to classify:", 
        answer_delimitator="### Correct answer :"
        ):
    """creates a custom output parser function based on specific delimitators"""

    def output_parser_func(outputs:list):
        """takes a decoded llm outputs as argument and returns input quote and predicted label"""
        def first_digit(sentence):
            match = re.search(r'\d', sentence)
            if match:
                return int(match.group())
            return -1
        
        # Ensure outputs is a list
        if not isinstance(outputs, list):
            outputs = [outputs]

        results = []
        for text in outputs:
            try:
                after_statement = text.split(statement_delimitator, 1)[1]
                quote, answer_part = after_statement.split(answer_delimitator, 1)
                quote = quote.strip()
                label = first_digit(answer_part)
                results.append((quote, label))
            except Exception as e:
                print(f"Parsing error for text: {text}\nError: {str(e)}")
                results.append((text.strip(), -1))
        return results
        
    return output_parser_func

output_parser_func = create_parser_function()


# test
# output_sample_1 = output_parser_func(formatted_sample_1['text'])
# output_sample_10 = output_parser_func(formatted_sample_10['text'])
# output_sample_1, output_sample_10