# -*- coding:utf-8 -*-
"""
@file name  : 02_predict_parse.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2024-02-26
@brief      : NER结果可视化
"""

import json


def extract_entities_from_lines(test_data_path, predictions_path):
    # Initialize lists to hold the data
    test_data_list = []
    predictions_list = []

    # Read test data line by line and parse JSON
    with open(test_data_path, 'r', encoding='utf-8') as test_data_file:
        for line in test_data_file:
            test_data_list.append(json.loads(line.strip()))

    # Read predictions line by line and parse JSON
    with open(predictions_path, 'r', encoding='utf-8') as predictions_file:
        for line in predictions_file:
            predictions_list.append(json.loads(line.strip()))

    # Ensure that the input data and predictions have the same number of entries
    if len(test_data_list) != len(predictions_list):
        raise ValueError("The number of entries in test data and predictions do not match.")

    # Extract entities for each pair of test data and predictions
    templates = []
    for test_data, predictions in zip(test_data_list, predictions_list):
        # Ensure that the input data and predictions have the same 'id'
        if test_data['id'] != predictions['id']:
            raise ValueError("The 'id' in test data and predictions do not match.")

        # Extract text and entities from predictions
        text = test_data['text']
        entities = predictions['entities']

        # Extract the entities from the text
        extracted_entities = []
        for entity in entities:
            entity_type, start, end = entity
            extracted_entity = text[start:end+1]
            extracted_entities.append((entity_type, extracted_entity))

        # Build the template string
        template = f"从文本\"{text}\"中，提取到实体"
        for entity_type, entity_text in extracted_entities:
            template += f" {entity_type} \"{entity_text}\"，"

        print(template)

    return templates


if __name__ == "__main__":
    # Test the function with the provided example
    path_test_data = r"G:\deep_learning_data\cluener_public\test.json"
    path_output_data = r"./outputsbert/test_prediction.json"

    extract_entities_from_lines(path_test_data, path_output_data)
