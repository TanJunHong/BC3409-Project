import nltk
import pandas
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, models


def nest_sentences(document: str) -> list[list[str]]:
    """
    Generates chunks of text from sentences <= 512 tokens
    @param document: Document to generate chunks from
    @return: A list with sentences tokenized.
    """
    nested: list[list[str]] = []
    sent: list[str] = []
    length: int = 0
    for sentence in nltk.sent_tokenize(text=document):
        length += len(sentence)
        if length < 512:
            sent.append(sentence)
        else:
            nested.append(sent)
            sent = [sentence]
            length = len(sentence)

    if sent:
        nested.append(sent)

    return nested


def generate_summary(nested_sentences: list[list[str]]) -> list[str]:
    """
    Generates summary of the nested sentences.
    @param nested_sentences: Nested sentences
    @return: A list with summary of the sentences
    """
    pretrained_path: str = "keith97/bert-small2bert-small-finetuned-cnn_daily_mail-summarization-newsroom-filtered"
    tokenizer: models.bert.tokenization_bert_fast.BertTokenizerFast = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=pretrained_path)
    model: models.encoder_decoder.modeling_encoder_decoder.EncoderDecoderModel = AutoModelForSeq2SeqLM.from_pretrained(
        pretrained_model_name_or_path=pretrained_path)

    summaries: list[list[str]] = []
    for nested in nested_sentences:
        if len(nested) > 0:
            input_tokenized: torch.Tensor = tokenizer.encode(text=" ".join(nested),
                                                             padding="max_length",
                                                             truncation=True,
                                                             max_length=512,
                                                             return_tensors="pt")
            summary_ids: torch.Tensor = model.generate(inputs=input_tokenized,
                                                       length_penalty=3.0,
                                                       min_length=30,
                                                       max_length=100)
            output: list[str] = [tokenizer.decode(token_ids=token_ids, skip_special_tokens=True) for token_ids in
                                 summary_ids]
            summaries.append(output)
        else:
            continue
    return [sentence for sublist in summaries for sentence in sublist]


def run_summariser(df: pandas.DataFrame) -> list:
    """
    Runs the summariser on the dataframe
    @param df: Dataframe to run summariser on
    @return: Summarised list
    """
    grouped_data: pandas.DataFrame = df.groupby(by="labels")["sentences"].apply(func="".join).reset_index()
    nested_sentences_lists: list[list[list[str]]] = [nest_sentences(document=sentences) for sentences in
                                                     grouped_data["sentences"]]
    summaries_list: list[list[str]] = [generate_summary(nested_sentences=nested_sentences)
                                       for nested_sentences in nested_sentences_lists]

    return summaries_list
