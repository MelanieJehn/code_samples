import my_utils
import transformers


if __name__ == '__main__':
    # Create Datasets
    # For DistilBERT:
    # model_class, tokenizer_class, pretrained_weights = (transformers.DistilBertModel, transformers.DistilBertTokenizer, 'distilbert-base-uncased')
    model_class, tokenizer_class, pretrained_weights = (transformers.BertModel, transformers.BertTokenizer, 'bert-base-uncased')
    # my_utils.create_token_dataset("/data", tokenizer_class)
    my_utils.create_context_dataset("/data", "classifier_dataset.csv", "inf_data_big.csv", tokenizer_class)
