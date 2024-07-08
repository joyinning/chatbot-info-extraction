
import gradio as gr
from model_utils import load_models, extract_information, predict_tags, extract_4w_qa, generate_why_or_how_question_and_answer

bert_model, bilstm_model, ner_tokenizer, id2label_ner = load_models()

def extract_and_display_info(user_input):
    if user_input:
        ner_tags = predict_tags(user_input, bilstm_model, ner_tokenizer, id2label_ner)
        extracted_info = extract_4w_qa(user_input, ner_tags)
        qa_result = generate_why_or_how_question_and_answer(extracted_info, user_input)

        if qa_result:
            extracted_info["Generated Question"] = qa_result["question"]
            extracted_info["Answer"] = qa_result["answer"]

        output_text = "Extracted Information:\n"
        for question, answer in extracted_info.items():
            output_text += f"- **{question}:** {answer}\n"
        return output_text
    else:
        return "Please enter some text."

iface = gr.Interface(
    fn=extract_and_display_info,
    inputs="text",
    outputs="text",
    title="Information Extraction Chatbot"
)
iface.launch()
