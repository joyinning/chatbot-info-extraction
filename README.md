# Information Extraction Chatbot
## Project Overview
This project aims to develop a sophisticated chatbot capable of extracting key information from text and answering questions in a 5W1H format. The project involves training and evaluating natural language processing (NLP) models and deploying them through a user-friendly Streamlit web interface.

## File Structure
```
app.py
model_utils.py
requirements.txt
models/
    ├── bert_model.pkl   
    ├── bilstm-model.pkl 
    ├── config.json   
    └── tokenizer/   
        ├── special_tokens_map.json
        ├── tokenizer_config.json
        └── tokenizer.json
        └── vocab.txt

```
1. `app.py`: The main script that defines the Gradio interface, handles user input, and interacts with functions in model_utils.py to perform information extraction and question answering.
2. `model_utils.py`: Contains utility functions for loading models, performing Named Entity Recognition (NER), extracting 4W (Who, What, When, Where) information, generating questions, and getting answers from GPT-3.5.
3. `requirements.txt`: Lists the Python packages required to run the application (e.g., torch, transformers, gradio, openai, tiktoken).
4. `models/`: A folder that stores the trained model files.
    - `bert_model.pkl`: The BERT model file.
    - `bilstm-model.pkl`: The BiLSTM model file.
    - `config.json`: The model configuration file.
    - `tokenizer/:` A folder containing tokenizer-related files.
        - `special_tokens_map.json`: Contains information about special tokens.
        - `tokenizer_config.json`: The tokenizer configuration file.
        - `tokenizer.json`: Contains tokenizer information.
        - `vocab.txt`: The vocabulary file.

## Phase 1: Data Preparation and Preprocessing
**Objective**: Prepare and process raw text data for model training and evaluation. <br>
**Process**:
1. **Data Acquisition**: Obtain a dataset containing labeled entities in sentences.
2. **Data Cleaning**: Remove irrelevant or noisy data points.
3. **Tokenization**: Split sentences into individual words or subwords.
4. **Label Encoding**: Convert entity labels into numerical representations.
5. **Train/Test Split**: Divide the data into training and testing sets.
6. **Tools**: pandas, matplotlib, transformers

**Key Results**:
1. The dataset consisted of 47,959 sentences with an average length of 125 words.
2. Defined a clear mapping for NER labels (e.g., "B-per" for beginning of a person's name).

## Phase 2: BERT Model Training (Named Entity Recognition)
**Objective**: Train a BERT model for accurate named entity recognition (NER). <br>
**Model Selection**: BERT was chosen for its ability to capture contextual information in text, crucial for identifying entities.

**Process**:

1. **Fine-tuning**: Fine-tune a pre-trained BERT model on the prepared NER dataset.
2. **Evaluation Metrics**: Utilize precision, recall, F1-score, and accuracy to assess model performance.
3. **Tools**: transformers, datasets, seqeval, PyTorch

**Key Results**:
1. Achieved 97.30% accuracy on the NER task, demonstrating the model's ability to identify entities effectively.
2. Successfully matching NER tags with both given sentences and unseen examples.
   
## Phase 3: BiLSTM Model with BERT Embeddings (Enhanced NER)
**Objective**: Combine a bidirectional LSTM (BiLSTM) model with BERT embeddings to further improve NER performance. <br>
**Model Selection**: BiLSTM was added to capture sequential dependencies in text, enhancing the model's understanding of entity relationships.

**Process**:
1. **Integration**: Incorporate a BiLSTM layer into the model architecture, utilizing BERT embeddings as input.
2. **Fine-tuning**: Continue fine-tuning the combined model to optimize its performance.

**Key Results**:
1. Observed a slight improvement in NER accuracy due to the BiLSTM's ability to model contextual information over longer sequences.
2. Successfully finding corresponding NER tags for given sentences and external examples.

## Phase 4: 5W1H Question Answering
**Objective**: Implement a system capable of answering "who," "what," "when," "where," "why," and "how" (5W1H) questions based on extracted information.<br>

**Process**:
1. **Information Extraction**: Utilize the BiLSTM-BERT model to extract named entities and their types from input text.
2. **4W Questions**: Use a question-answering model to answer "who," "what," "when," and "where" questions directly from the extracted information.
3. **1W1H Questions**: Generate "why" and "how" questions based on the extracted information and use GPT-3.5 to provide insightful answers.
4. **Tools**: transformers, deepset/bert-base-cased-squad2, OpenAI GPT-3.5

**Key Results**:
1. The system successfully answered 5W1H questions, showcasing its ability to understand and extract information from text comprehensively.

## Phase 5: Deployment with Streamlit
**Objective**: Deploy the chatbot as a user-friendly web application. <br>

**Process**:

1. **File Structure**: Organize code into app.py, model_utils.py, requirements.txt, and models/ directory.
2. **Streamlit Interface**: Create an intuitive interface for users to input text and receive extracted information.
3. **Deployment**: Deploy the application to Streamlit Cloud or other platforms like Heroku or AWS.
4. **Tools**: Streamlit

**Key Results**:
1. The chatbot is readily accessible through a web interface, allowing users to interact with it effortlessly.
   
## Conclusion
This project successfully developed an information extraction chatbot capable of understanding text, identifying entities, and answering 5W1H questions. The combination of BERT, BiLSTM, and GPT-3.5 models, along with a user-friendly Streamlit interface, demonstrates the power of NLP for extracting valuable insights from text data.
