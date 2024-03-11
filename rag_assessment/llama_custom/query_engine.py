from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core import PromptTemplate
from llama_index.llms.llama_cpp import LlamaCPP
from typing import Any, List, Optional

from llama_index.core.base.response.schema import Response
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.llms.llm import LLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.response_synthesizers import (
    BaseSynthesizer,
    ResponseMode,
    get_response_synthesizer,
)
from llama_index.core.service_context import ServiceContext
from llama_index.llms.ai21 import AI21


class RAGQueryEngine(CustomQueryEngine):
    """RAG Query Engine."""

    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)
        response_obj = self.response_synthesizer.synthesize(query_str, nodes)
        return response_obj

    @classmethod
    def from_args(
            cls,
            retriever: BaseRetriever,
            llm: LLM,
            response_synthesizer: Optional[BaseSynthesizer] = None,
            node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
            # response synthesizer args
            response_mode: ResponseMode = ResponseMode.COMPACT,
            text_qa_template: Optional[BasePromptTemplate] = None,
            refine_template: Optional[BasePromptTemplate] = None,
            summary_template: Optional[BasePromptTemplate] = None,
            simple_template: Optional[BasePromptTemplate] = None,
            output_cls: Optional[BaseModel] = None,
            use_async: bool = False,
            streaming: bool = False,
            # deprecated
            service_context: Optional[ServiceContext] = None,
            **kwargs: Any,
    ) -> "RAGQueryEngine":
        """Initialize a RAGQueryEngine object.".

        Args:
            retriever (BaseRetriever): A retriever object.
            service_context (Optional[ServiceContext]): A ServiceContext object.
            node_postprocessors (Optional[List[BaseNodePostprocessor]]): A list of
                node postprocessors.
            verbose (bool): Whether to print out debug info.
            response_mode (ResponseMode): A ResponseMode object.
            text_qa_template (Optional[BasePromptTemplate]): A BasePromptTemplate
                object.
            refine_template (Optional[BasePromptTemplate]): A BasePromptTemplate object.
            simple_template (Optional[BasePromptTemplate]): A BasePromptTemplate object.

            use_async (bool): Whether to use async.
            streaming (bool): Whether to use streaming.
            optimizer (Optional[BaseTokenUsageOptimizer]): A BaseTokenUsageOptimizer
                object.

        """
        llm = llm

        response_synthesizer = response_synthesizer or get_response_synthesizer(
            llm=llm,
            service_context=service_context,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            summary_template=summary_template,
            simple_template=simple_template,
            response_mode=response_mode,
            output_cls=output_cls,
            use_async=use_async,
            streaming=streaming,
        )

        return cls(
            retriever=retriever,
            response_synthesizer=response_synthesizer
        )


class RAGStringQueryEngine(CustomQueryEngine):
    """RAG String Query Engine. Incorporates context in Response"""

    retriever: BaseRetriever
    #llm: LlamaCPP
    llm: AI21
    qa_prompt: PromptTemplate

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)

        response_txt, fmt_qa_prompt = self.generate_metadata_response(
            nodes,
            query_str
        )
        response_str = response_txt + "\n\n" + fmt_qa_prompt
        response_obj = Response(response_str.lstrip(), nodes)
        return response_obj

    def generate_context_response(self, retrieved_nodes, query_str):
        context_str = "\n\n".join([r.get_content(metadata_mode="all") for r in retrieved_nodes])
        fmt_qa_prompt = self.qa_prompt.format(
            context_str=context_str, query_str=query_str
        )
        response = self.llm.complete(fmt_qa_prompt)
        return str(response), fmt_qa_prompt

    def generate_metadata_response(self, retrieved_nodes, query_str):
        context_str = "\n\n".join([r.get_content(metadata_mode="all") for r in retrieved_nodes])
        fmt_qa_prompt = self.qa_prompt.format(
            context_str=context_str, query_str=query_str
        )
        response = self.llm.complete(fmt_qa_prompt)
        new_fmt_qa_prompt = self.extract_metadata(retrieved_nodes, fmt_qa_prompt)
        return str(response), new_fmt_qa_prompt

    def extract_metadata(self, nodes, prompt):
        # llama_db: {'total_pages': 77, 'file_path': './data/llama2.pdf', 'source': '36'}
        # textai_vector: {'page_label':, 'file_name', 'file_path' ...}
        a_prompt = PromptTemplate(
            "Context information is below.\n"
            "---------------------\n"
            "{metadata_str}\n"
            "---------------------\n"
        )
        metadata_str_list = []
        for node in nodes:
            metadata_dict = node.metadata
            node_context_sents = node.get_content().split(".")
            cur_str = ""
            for key in metadata_dict:
                cur_str += f"{key}: {metadata_dict[key]}\n"
            if len(node_context_sents) > 1:
                cur_str += ".".join(node_context_sents[:2])
            metadata_str_list.append(cur_str)
        metadata_str = "\n\n".join(metadata_str_list)
        fmt_qa_prompt = a_prompt.format(metadata_str=metadata_str)
        return fmt_qa_prompt
