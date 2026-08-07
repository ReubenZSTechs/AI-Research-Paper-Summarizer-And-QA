from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

import yaml
import logging
import os

from backend.core.decorators.log_llm_calls import log_llm_calls, log_output_call

logger = logging.getLogger(__name__)


CONFIG = {
    "MODEL_BRIDGE": {
        "llm": os.getenv("LLM_BRIDGE_URL", "http://LLM:15000/v1"),
        "embedding": os.getenv("EMBEDDING_BRIDGE_URL", "http://embedding:15001/v1"),
        "reranker": os.getenv("RERANKER_BRIDGE_URL", "http://reranker:15002/v1"),
    },
    "SERVED_MODEL_NAME": {
        "llm": "llm-server",
        "embedding": "embedding-server",
        "reranker": "reranker-server",
    },
}


class AgentReader:
    def read_config(self, config: str) -> dict:
        try:
            with open(config, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)

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
            self.agent_id = self.agent_config["agent"]
            missing = [key for key in ("model_type", "role") if key not in self.agent_id]
            if missing:
                raise KeyError(f"agent section missing: {missing}")

            self.agent_system_prompt = self.agent_config["system_prompt"]
            self.agent_generation_config = self.agent_config["generation"]

        except KeyError as e:
            print(f"Missing key in yaml configuration:\n\n{e.args[0]}")
            raise

        model_type = self.agent_id["model_type"]
        self.bridge_url = CONFIG["MODEL_BRIDGE"].get(model_type)
        self.served_model_name = CONFIG["SERVED_MODEL_NAME"].get(model_type)

        if not self.bridge_url or not self.served_model_name:
            raise ValueError(f"No vLLM container bridge available for {self.agent_id['role']}")

        self.client = OpenAI(base_url=self.bridge_url, api_key="not-needed")
        logger.info(f"Initiated agent with role {self.agent_id['role']}")


    def format_msg_payload(self, user_prompt):
        return [
            {"role": "system", "content": self.agent_system_prompt},
            {"role": "user", "content": user_prompt},
        ]


    def format_generation_args(self):
        config = self.agent_generation_config
        args = {
            "max_tokens": config["max_tokens"],
            "temperature": config.get("temperature", 0.1),
            "top_p": config.get("top_p", 0.9),
            "frequency_penalty": config.get("frequency_penalty", 0.0),
            "presence_penalty": config.get("presence_penalty", 0.0),
        }

        return args


    @log_llm_calls
    def generate_response(self, user_prompt, extra_args=None):
        msg_payload = self.format_msg_payload(user_prompt=user_prompt)
        generation_settings = self.format_generation_args()

        response = self.client.chat.completions.create(
            model=self.served_model_name,
            messages=msg_payload,
            extra_body=extra_args or {},
            **generation_settings,
        )

        self.last_usage = response.usage
        return response.choices[0].message.content


    @log_output_call
    def generate_answer(self, user_prompt, extra_args=None):
        msg_payload = self.format_msg_payload(user_prompt=user_prompt)
        generation_settings = self.format_generation_args()

        stream = self.client.chat.completions.create(
            model=self.served_model_name,
            messages=msg_payload,
            stream=True,
            stream_options={"include_usage": True},
            extra_body=extra_args or {},
            **generation_settings,
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