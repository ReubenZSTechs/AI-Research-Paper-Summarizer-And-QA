from dotenv import load_dotenv
load_dotenv()

from transformers import AutoTokenizer, AutoModelForTokenClassification
from openai import OpenAI

import yaml
import logging
import os

from backend.core.decorators.log_llm_calls import log_llm_calls, log_output_call

logger = logging.getLogger(__name__)



CONFIG = {
    'MODEL_BRIDGE':{
        'llm': 'http://localhost:15000/v1',
        'embedding': 'http://localhost:15001/v1',
        'reranker': 'http://localhost:15002/v1'
    }
}


class AgentReader:
    def __init__(self):
        pass

    def read_config(self, config: str) -> dict:
        try:
            with open(config, "r", encoding='utf-8') as f:
                agent_config = yaml.safe_load(f)

            return agent_config
        
        except FileNotFoundError as e:
            print(f"YAML config not found:\n\n{e}")
            return {}

        except Exception as e:
            print(f"Another error occured:\n\n{e}")
            return {}
        

class Agent:
    def __init__(self, yaml_config: str):
        self.reader = AgentReader()
        self.agent_config = self.reader.read_config(yaml_config)

        try:
            self.agent_id = self.agent_config['agent']

            assert ['model_type', 'role'] in self.agent_id.keys()

            self.agent_system_prompt = self.agent_config['system_prompt']
            self.agent_generation_config = self.agent_config['generation']
        
        except KeyError as e:
            print(f"Missing key in yaml configuration:\n\n{e.args[0]}")
            raise

        self.bridge_url = CONFIG['MODEL_BRIDGE'].get(self.agent_id['model_type'], None)

        if not self.bridge_url:
            raise ValueError(f"No vLLM container url bridge available for {self.agent_id['role']}")
        
        self.client = OpenAI(base_url=self.bridge_url, api_key='not-needed')
        logger.info(f"Initiated agent with role {self.agent_id['role']}")


    def format_msg_payload(self, user_prompt): # TODO Add context agent manager for context management on system prompt and user prompt
        agent_input = [
            {
                'role': 'system',
                'content': self.agent_system_prompt
            },
            {
                'role': 'user',
                'content': user_prompt
            }
        ]

        return agent_input


    def format_generation_args(self):
        # TODO Find more generation arguments that is OpenAI compatible
        generation_args = {
            'max_tokens': self.agent_generation_config['max_tokens'], # TODO Remove max_tokens when agent context manager is built
            'temperature': self.agent_generation_config.get('temperature', 0.1),
            'top_p': self.agent_generation_config.get('top_p', 0.9),
            'frequency_penalty': self.agent_generation_config.get("frequency_penalty", 1.0)
        }

        return generation_args
    

    @log_llm_calls
    def generate_response(self, user_prompt, extra_args=None):
        msg_payload = self.format_msg_payload(user_prompt=user_prompt)
        generation_settings = self.format_generation_args()

        response = self.client.chat.completions.create(
            model = self.bridge_url,
            messages=msg_payload,
            extra_body=extra_args or {},
            **generation_settings
        )

        self.last_usage = response.usage

        return response.choices[0].message.content
    

    @log_output_call
    def generate_answer(self, user_prompt, extra_args=None):
        msg_payload = self.format_msg_payload(user_prompt=user_prompt)
        generation_settings = self.format_generation_args()

        stream = self.client.chat.completions.create(
            model=self.bridge_url,
            messages=msg_payload,
            stream=True,
            stream_options={'include_usage': True},
            extra_body=extra_args or {},
            **generation_settings
        )

        for chunk in stream:
            if chunk.usage:
                self.last_usage = chunk.usage

            if chunk.choices:
                delta = chunk.choices[0].delta.content
            else:
                delta = None

            if delta:
                yield delta