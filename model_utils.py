import torch
from transformers import BertTokenizer, AutoModelForTokenClassification, pipeline
import pickle # for saving and loading Python objects
from openai import OpenAI
import tiktoken
from transformers import AutoConfig, AutoTokenizer
import os
import torch.nn as nn
from transformers import AutoModel, AutoConfig

client = OpenAI(api_key="")

# Define BiLSTMForTokenClassification Class


class BiLSTMForTokenClassification(nn.Module):
    """
        This model combines BERT embeddings with a Bidirectional LSTM (BiLSTM) for token-level classification
        tasks like Named Entity Recognition (NER).

        Args:
            pretrained_model_name_or_path: Name of the pre-trained BERT model to use (e.g., "bert-base-cased").
            num_labels: Number of different labels to predict.
            hidden_size: Dimension of the hidden states in the BiLSTM (default: 128).
            num_lstm_layers: Number of stacked BiLSTM layers (default: 1).
    """
    def __init__(self, model_name, num_labels, hidden_size=128, num_lstm_layers=1):
        super().__init__()
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        # Freeze BERT embeddings
        for name, param in self.bert.named_parameters():
            if name.startswith("embeddings"):
                param.requires_grad = False

        self.bilstm = nn.LSTM(self.bert.config.hidden_size, hidden_size, num_layers=num_lstm_layers, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        lstm_output, _ = self.bilstm(sequence_output)
        lstm_output = self.dropout(lstm_output)

        logits = self.classifier(lstm_output)
        loss = None

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels))
            valid_mask = (active_labels >= 0) & (active_labels < self.num_labels)
            active_logits = active_logits[valid_mask]
            active_labels = active_labels[valid_mask]
            loss = loss_fct(active_logits, active_labels)

        return {'loss': loss, 'logits': logits}

def load_models():
    """
    Loads the custom BiLSTM NER model and tokenizer from local files.

    Returns:
        model (BiLSTMForTokenClassification): The loaded BiLSTM NER model.
        tokenizer (AutoTokenizer): The loaded tokenizer associated with the model.
        id2label_ner (dict): A dictionary mapping numerical label indices to NER tags.
    """

    model_dir = "./models/bilstm_ner"
    tokenizer_dir = "./models/tokenizer"

    id2label_ner = {
        0: 'O', 1: 'I-art', 2: 'B-org', 3: 'B-geo', 4: 'I-per', 5: 'B-eve',
        6: 'I-geo', 7: 'B-per', 8: 'I-nat', 9: 'B-art', 10: 'B-tim', 11: 'I-gpe',
        12: 'I-tim', 13: 'B-nat', 14: 'B-gpe', 15: 'I-org', 16: 'I-eve'
    }

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    config.id2label = id2label_ner
    config.num_labels = len(id2label_ner)

    model = BiLSTMForTokenClassification(model_name=config._name_or_path, num_labels=config.num_labels)
    model.config.id2label = id2label_ner
    model.load_state_dict(torch.load(os.path.join(model_dir, 'pytorch_model.bin'), map_location=torch.device('cpu')))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)

    return model, tokenizer, id2label_ner

# QA model
qa_model = pipeline('question-answering', model='deepset/bert-base-cased-squad2')

# Function to extract information
def extract_information(text, bert_model, bilstm_model, ner_tokenizer, id2label_ner):
    """
    Extracts information from the given text using NER tags and generates 'Why' or 'How' questions with answers.

    Args:
        text: The input text string.
        bert_model: The pre-trained BERT model for token classification.
        bilstm_model: The BiLSTM model for NER tag prediction.
        ner_tokenizer: The tokenizer for the BiLSTM model.
        id2label_ner: A dictionary mapping numerical label indices to NER tags.

    Returns:
        A dictionary containing extracted 4W information, generated question, and answer.
    """
    extracted_info = {}

    ner_tags = predict_tags(text, bilstm_model, ner_tokenizer, id2label_ner)
    
    extracted_info.update(extract_4w_qa(text, ner_tags))
    
    qa_result = generate_why_or_how_question_and_answer(extracted_info, text)
    if qa_result:
        extracted_info.update(qa_result)
        prompt = f"Question: {qa_result['question']}\nContext: {text}\nAnswer:"
        extracted_info["Token Count"] = count_tokens(prompt)
        
    return extracted_info


def predict_tags(sentence, model, tokenizer, label_map):
    """
    Predicts NER tags for a given sentence using the specified model and tokenizer.

    Args:
        sentence (str): The input sentence.
        model (nn.Module): The NER model.
        tokenizer: The tokenizer used for the model.
        label_map (dict): A dictionary mapping numerical label indices to their corresponding tags.

    Returns:
        list: A list of predicted tags for each token in the sentence.
    """
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sentence)))
    inputs = tokenizer.encode(sentence, return_tensors='pt')

    outputs = model(inputs)
    logits = outputs['logits']
    predictions = torch.argmax(logits, dim=2)

    labels = [label_map.get(prediction.item(), "O") for prediction in predictions[0][1:-1]]
    return labels

def extract_4w_qa(sentence, ner_tags):
    """
    Extracts 4w (Who, What, When, Where) information from a sentence
    using NER tags and a question-answering model.

    Args:
        sentence: The input sentence as a string.
        ner_tags: A list of predicted NER tags for each token in the sentence.

    Returns:
        A dictionary where keys are 5W1H question words and values are the corresponding
        answers extracted from the sentence.
    """
    result = {}
    questions = {
        "B-per": "Who",
        "I-per": "Who",
        "B-geo": "Where",
        "I-geo": "Where",
        "B-org": "What organization",
        "I-org": "What organization",
        "B-tim": "When",
        "I-tim": "When",
        "B-art": "What art",
        "I-art": "What art",
        "B-eve": "What event",
        "I-eve": "What event",
        "B-nat": "What natural phenomenon",
        "I-nat": "What natural phenomenon",
    }

    for ner_tag, entity in zip(ner_tags, sentence.split()):  # Removed pos_tags
        if ner_tag in questions:
            question = f"{questions[ner_tag]} is {entity}?"  # Removed pos_tag
            answer = qa_model(question=question, context=sentence)["answer"]
            result[questions[ner_tag]] = answer

    return result

def count_tokens(text):
    """
    Counts the number of tokens in a text string using the tiktoken encoding for GPT-3.5 Turbo.

    Args:
        text: The input text string.

    Returns:
        The number of tokens in the text.
    """
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

def generate_why_or_how_question_and_answer(extracted_info, sentence):
    """
    Generates a "Why" or "How" question based on the extracted 4W information and gets the answer using GPT-3.5.

    Args:
        extracted_info: A dictionary containing the extracted 4W information.
        sentence: The original sentence.

    Returns:
        A dictionary containing the generated question and its answer, or None if no relevant question can be generated.
    """

    prompt_template = """
    Given the following extracted information and the original sentence, generate a relevant "Why" or "How" question and provide a concise answer based on the given context.

    Extracted Information: {extracted_info}
    Sentence: {sentence}

    Question and Answer:
    """

    prompt = prompt_template.format(extracted_info=extracted_info, sentence=sentence)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=150,
        stop=None,
        temperature=0.5,
    )

    question_and_answer = response.choices[0].message.content.strip()

    if question_and_answer:
        try:
            question, answer = question_and_answer.split("\n", 1)
            return {"question": question, "answer": answer}
        except ValueError:
            return None
    else:
        return None

