from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    CorrectnessEvaluator,
    RelevancyEvaluator,
    BatchEvalRunner,
)
from llama_index.core.llama_dataset import (
    LabelledRagDataset,
)

from data_processing.vector_database import (
    load_vector_database,
)
from llama_custom.rag_application import (
    get_embed_model,
    load_llm_model_cloud,
    get_query_engine,
    load_llm_model,
)
from llama_custom.retriever import VectorDBRetriever

from evaluation.correctness import CorrectnessEvaluator

import asyncio
import argparse
import json
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).split('/')
ROOT_DIR = ('/').join(ROOT_DIR[:-1])

api_key = "XcXwkV8LXRtuoqZdjFWIwS3fdGx1fS83"

parser = argparse.ArgumentParser()
parser.add_argument('--db', default=f'{ROOT_DIR}/evaluation/truthful_db', type=str)
parser.add_argument('--json_path', default=f'{ROOT_DIR}/evaluation/data/rag_dataset.json', type=str)
parser.add_argument('--engine', default='rag_citations', type=str)
parser.add_argument('--eval', default='llm', type=str)
parser.add_argument('--out', default='./eval_rag', type=str)
parser.add_argument('--local', default=False, type=bool)


async def evaluate_llm(query_engine, rag_dataset, llm, evaluators=["faithfulness"]):
    evaluator_dict = {}
    if "faithfulness" in evaluators:
        faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
        evaluator_dict["faithfulness"] = faithfulness_evaluator
    if "correctness" in evaluators:
        # defaults to 4.
        correctness_evaluator = CorrectnessEvaluator(llm=llm)
        evaluator_dict["correctness"] = correctness_evaluator
    if "relevancy" in evaluators:
        relevancy_evaluator = RelevancyEvaluator(llm=llm)
        evaluator_dict["relevancy"] = relevancy_evaluator

    # Run multiple evaluators on a set of questions
    # this contains 152 questions!
    questions = [e.query for e in rag_dataset.examples]
    questions = questions[:10]

    runner = BatchEvalRunner(
        evaluator_dict,
        workers=8,
    )

    eval_results = await runner.aevaluate_queries(
        query_engine, queries=questions
    )

    return eval_results


def evaluate_llm_single_query(query_engine, llm):
    correctness_evaluator = FaithfulnessEvaluator(llm=llm)
    query = (
        "Are vampires real?"
    )

    response = query_engine.query(query)
    reference = "No, vampires are not real"
    contexts = [str(cont) for cont in response.source_nodes]
    # faithfulness uses contexts and correctness uses the reference
    eval_result = correctness_evaluator.evaluate(
        query=query,
        response=str(response).lstrip(),
        contexts=contexts,
        reference=reference
    )
    return eval_result


def retrieve_eval_info(eval_results, path):
    """
    Print mean score per metric used for "eval_results" and write contents of "eval_results" to "path"
    :param eval_results: a dictionary containing a list of EvaluationResponse per key/metric
    :param path: the evaluation results will be saved here
    :return:
    """
    out_dict = {}
    for i, res in enumerate(eval_results["relevancy"]):
        out_dict[i] = {}
        out_dict[i]['query'] = res.query
        out_dict[i]['response'] = res.response
        out_dict[i]['contexts'] = res.contexts

    for key in eval_results:
        eval_list = eval_results[key]
        score_list = []
        feedback_list = []
        for i, el in enumerate(eval_list):
            out_dict[i][key] = {}
            out_dict[i][key]['score'] = el.score
            out_dict[i][key]['passing'] = el.passing
            out_dict[i][key]['feedback'] = el.feedback
            if key != "correctness" or el.feedback != "unknown":
                score_list.append(el.score)
            feedback_list.append(el.feedback)
        if len(score_list) > 0:
            score = sum(score_list) / len(score_list)
        else:
            score = -1.
        print(f'{key}: score {score}')
    with open(path + '.txt', 'w') as file:
        file.write(json.dumps(out_dict))


def data_setup(db_name, dataset_path):
    rag_dataset = LabelledRagDataset.from_json(dataset_path)
    vector_store = load_vector_database(db_name)
    return vector_store, rag_dataset


def get_rag_components(vector_store, engine, local):
    embed_model = get_embed_model()
    if local:
        # model_path='../llama_index/models/llama-2-13b-chat.Q4_0.gguf'
        llm = load_llm_model()
    else:
        llm = load_llm_model_cloud()
    retriever = VectorDBRetriever(
        vector_store, embed_model, query_mode="default", similarity_top_k=2
    )
    query_engine = get_query_engine(engine, retriever, llm)
    return embed_model, llm, retriever, query_engine


def evaluate_performance(query_engine, llm, rag_dataset, out_path, eval_type='llm'):
    # correctness: relevance and correctness of a generated answer against a reference answer (factual correctness)
    # faithfulness: measure if the response from a query engine matches any source nodes (helps with hallucination)
    # relevancy: measure if the response + source nodes match the query
    if eval_type == 'llm':
        eval_results = asyncio.run(evaluate_llm(query_engine, rag_dataset, llm,
                                                ["faithfulness", "correctness", "relevancy"]))
        retrieve_eval_info(eval_results=eval_results, path=out_path)
    else:
        eval_result = evaluate_llm_single_query(query_engine, llm)
        print(eval_result)


def main():
    args = parser.parse_args()
    vector_store, rag_dataset = data_setup(args.db, args.json_path)
    embed_model, llm, retriever, query_engine = get_rag_components(vector_store, args.engine, args.local)
    evaluate_performance(query_engine, llm, rag_dataset, args.out, eval_type=args.eval)

    # performance using rag_engine:
    # faithfulness: score 0.0
    # correctness: score 1.0
    # relevancy: score 0.0


if __name__ == "__main__":
    main()
