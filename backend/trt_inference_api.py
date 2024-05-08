import json
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer

import tensorrt_llm
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import ModelRunner
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner

import numpy as np
import torch 

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp

class ModelProcessor:
    def __init__(self,engine_dir,tokenizer_dir):

        self.token_path = "/mistral-7b-int4-chat_1.2/mistral7b_hf_tokenizer"
        self.engine_path = "/mistral-7b-int4-chat_1.2/trt_engines"
        self.max_output_len = 1024
        self.max_attention_window_size = None
        self.sink_token_length = None
        self.log_level = 'error'
        self.engine_dir = engine_dir
        self.use_py_session = True
        self.input_text = ["Do you have any idea about sheffield hallam university"]
        self.use_prompt_template = True
        self.input_file = None
        self.max_input_length = 1024
        self.output_csv = None
        self.output_npy = None
        self.output_logits_npy = None
        self.tokenizer_dir = tokenizer_dir
        self.tokenizer_type = None
        self.vocab_file = None
        self.num_beams = 1
        self.temperature = 1.0
        self.top_k = 1
        self.top_p = 0.0
        self.length_penalty = 1.0
        self.repetition_penalty = 1.0
        self.presence_penalty = 0.0
        self.frequency_penalty = 0.0
        self.debug_mode = False
        self.add_special_tokens = True
        self.streaming = False
        self.streaming_interval = 5
        self.prompt_table_path = None
        self.prompt_tasks = None
        self.lora_dir = None
        self.lora_task_uids = None
        self.lora_ckpt_source = "hf"
        self.num_prepend_vtokens = None
        self.run_profiling = False
        self.medusa_choices = None


        self.DEFAULT_PROMPT_TEMPLATES = {
            'InternLMForCausalLM':
            "<|User|>:{input_text}<eoh>\n<|Bot|>:",
            'qwen':
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n",
        }
        
        self.tokenizer = None
        self.pad_id = None
        self.end_id = None


        
        self.model_name = None
        self.model_version = None
        self.log_level = 'error'

        self.stop_words_list = None
        self.bad_words_list = None

     

        self.runtime_rank = tensorrt_llm.mpi_rank()
        logger.set_level(self.log_level)
       
        if self.tokenizer_dir is None:
            logger.warning(
                "tokenizer_dir is not specified. Try to infer from model_name, but this may be incorrect."
            )

        # read model 
        self.read_model_name(self.engine_dir)
        # load the tokenizer
        self.load_tokenizer(self.tokenizer_dir, self.vocab_file, self.tokenizer_type)

        self.prompt_template = None
        if self.use_prompt_template and self.model_name in self.DEFAULT_PROMPT_TEMPLATES:
            self.prompt_template = self.DEFAULT_PROMPT_TEMPLATES[self.model_name]

        self.batch_input_ids =  self.parse_input(
                                input_text=self.input_text,
                                prompt_template=self.prompt_template,
                                input_file=self.input_file,
                                add_special_tokens=self.add_special_tokens,
                                max_input_length=self.max_input_length,
                                pad_id=self.pad_id,
                                num_prepend_vtokens=self.num_prepend_vtokens
                                )
        
        self.input_lengths = [x.size(0) for x in self.batch_input_ids]


        self.runner_cls = ModelRunner if self.use_py_session else ModelRunnerCpp
        self.runner_kwargs = dict(engine_dir=self.engine_dir,
                        lora_dir=self.lora_dir,
                        rank=self.runtime_rank,
                        debug_mode=self.debug_mode,
                        lora_ckpt_source=self.lora_ckpt_source)
    
        self.runner = self.runner_cls.from_dir(**self.runner_kwargs)

    def runINference(self,input_text):
        
        self.input_text = [input_text]
        #self.input_text = input_text
        self.batch_input_ids =  self.parse_input(
                                input_text=self.input_text,
                                prompt_template=self.prompt_template,
                                input_file=self.input_file,
                                add_special_tokens=self.add_special_tokens,
                                max_input_length=self.max_input_length,
                                pad_id=self.pad_id,
                                num_prepend_vtokens=self.num_prepend_vtokens
                                )
        
        self.input_lengths = [x.size(0) for x in self.batch_input_ids]

        with torch.no_grad():
            self.outputs = self.runner.generate(
                self.batch_input_ids,
                max_new_tokens=self.max_output_len,
                max_attention_window_size=self.max_attention_window_size,
                sink_token_length=self.sink_token_length,
                end_id=self.end_id,
                pad_id=self.pad_id,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                num_beams=self.num_beams,
                length_penalty=self.length_penalty,
                repetition_penalty=self.repetition_penalty,
                presence_penalty=self.presence_penalty,
                frequency_penalty=self.frequency_penalty,
                stop_words_list=self.stop_words_list,
                bad_words_list=self.bad_words_list,
                lora_uids=self.lora_task_uids,
                prompt_table_path=self.prompt_table_path,
                prompt_tasks=self.prompt_tasks,
                streaming=self.streaming,
                output_sequence_lengths=True,
                return_dict=True,
                medusa_choices=self.medusa_choices)
            torch.cuda.synchronize()

        response = ""
        if self.runtime_rank == 0:
            output_ids = self.outputs['output_ids']
            sequence_lengths = self.outputs['sequence_lengths']
            response = self.print_output(
                            output_ids,
                            self.input_lengths,
                            sequence_lengths,
                            output_csv=self.output_csv,
                            output_npy=self.output_npy,
                            )

        return response

    def read_model_name(self,engine_dir: str):
        engine_version = tensorrt_llm.runtime.engine.get_engine_version(engine_dir)

        with open(Path(engine_dir) / "config.json", 'r') as f:
            config = json.load(f)

        if engine_version is None:
            return config['builder_config']['name'], None

        self.model_name = config['pretrained_config']['architecture']
        self.model_version = None
        
    def load_tokenizer(self, 
                       tokenizer_dir: Optional[str] = None,
                       vocab_file: Optional[str] = None,
                       tokenizer_type: Optional[str] = None):
        if vocab_file is None:
            use_fast = True
            if tokenizer_type is not None and tokenizer_type == "llama":
                use_fast = False
            # Should set both padding_side and truncation_side to be 'left'
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir,
                                                            legacy=False,
                                                            padding_side='left',
                                                            truncation_side='left',
                                                            trust_remote_code=True,
                                                            tokenizer_type=tokenizer_type,
                                                            use_fast=use_fast)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id
        self.end_id = self.tokenizer.eos_token_id

    def parse_input(self, 
                    input_text=None,
                    prompt_template=None,
                    input_file=None,
                    add_special_tokens=True,
                    max_input_length=923,
                    pad_id=None,
                    num_prepend_vtokens=[]
                    ):
        if pad_id is None:
            pad_id = self.pad_id

        batch_input_ids = []
        if input_file is None:
            for curr_text in input_text:
                if prompt_template is not None:
                    curr_text = prompt_template.format(input_text=curr_text)
                input_ids = self.tokenizer.encode(curr_text,
                                                   add_special_tokens=add_special_tokens,
                                                   truncation=True,
                                                   max_length=max_input_length)
                batch_input_ids.append(input_ids)

        if num_prepend_vtokens:
            assert len(num_prepend_vtokens) == len(batch_input_ids)
            base_vocab_size = self.tokenizer.vocab_size - len(
                self.tokenizer.special_tokens_map.get('additional_special_tokens', []))
            for i, length in enumerate(num_prepend_vtokens):
                batch_input_ids[i] = list(
                    range(base_vocab_size,
                          base_vocab_size + length)) + batch_input_ids[i]

        batch_input_ids = [
            torch.tensor(x, dtype=torch.int32) for x in batch_input_ids
        ]
        return batch_input_ids

    def print_output(self, output_ids,
                     input_lengths,
                     sequence_lengths,
                     output_csv=None,
                     output_npy=None,
                     ):
        response = ""
        batch_size, num_beams, _ = output_ids.size()
        if output_csv is None and output_npy is None:
            for batch_idx in range(batch_size):
                inputs = output_ids[batch_idx][0][:input_lengths[batch_idx]].tolist()
                input_text = self.tokenizer.decode(inputs)
                print(f'Input [Text {batch_idx}]: \"{input_text}\"')
                for beam in range(num_beams):
                    output_begin = input_lengths[batch_idx]
                    output_end = sequence_lengths[batch_idx][beam]
                    outputs = output_ids[batch_idx][beam][output_begin:output_end].tolist()
                    output_text = self.tokenizer.decode(outputs)
                    print(f'Output [Text {batch_idx} Beam {beam}]: \"{output_text}\"')
                    response = response + " " + output_text + '\n'

        output_ids = output_ids.reshape((-1, output_ids.size(2)))
        return response

# Then call parse_input and print_output as needed


from typing import Any, Dict, Iterator, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk

class CustomLLM(LLM):
    """A custom chat model that echoes the first `n` characters of the input.

    When contributing an implementation to LangChain, carefully document
    the model including the initialization parameters, include
    an example of how to initialize the model and include any relevant
    links to the underlying models documentation or API.

    Example:

        .. code-block:: python

            model = CustomChatModel(n=2)
            result = model.invoke([HumanMessage(content="hello")])
            result = model.batch([[HumanMessage(content="hello")],
                                 [HumanMessage(content="world")]])
    """

    n: int
    proccessor: Any = None
    engine_dir: str
    tokenizer_dir: str

    """The number of characters from the last message of the prompt to be echoed."""



    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        print(self.engine_dir)
        print(self.tokenizer_dir)
        self.n = 10
        self.proccessor = ModelProcessor(engine_dir=self.engine_dir, tokenizer_dir=self.tokenizer_dir)


        

    def initiate(self):
        pass

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input.

        Override this method to implement the LLM logic.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            The model output as a string. Actual completions SHOULD NOT include the prompt.
        """

        
        resp = self.proccessor.runINference(prompt)

        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return resp

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": "CustomChatModel",
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "custom"