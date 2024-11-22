import copy
import os
import time
import warnings
from functools import partial
from typing import Any

import requests
import tiktoken

from openhands.core.config import LLMConfig
from openhands.core.logger import openhands_logger as logger

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import litellm
from litellm import Message as LiteLLMMessage
from litellm import ModelInfo, PromptTokensDetails
from litellm import completion as litellm_completion
from litellm import completion_cost as litellm_completion_cost
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
)
from litellm.types.utils import CostPerToken, ModelResponse, Usage

from openhands.core.exceptions import CloudFlareBlockageError
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message
from openhands.llm.debug_mixin import DebugMixin
from openhands.llm.fn_call_converter import (
    STOP_WORDS,
    convert_fncall_messages_to_non_fncall_messages,
    convert_non_fncall_messages_to_fncall_messages,
)
from openhands.llm.metrics import Metrics
from openhands.llm.retry_mixin import RetryMixin

__all__ = ['LLM']

# tuple of exceptions to retry on
LLM_RETRY_EXCEPTIONS: tuple[type[Exception], ...] = (
    APIConnectionError,
    # FIXME: APIError is useful on 502 from a proxy for example,
    # but it also retries on other errors that are permanent
    APIError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
)

# cache prompt supporting models
# remove this when we gemini and deepseek are supported
CACHE_PROMPT_SUPPORTED_MODELS = [
    'claude-3-5-sonnet-20241022',
    'claude-3-5-sonnet-20240620',
    'claude-3-5-haiku-20241022',
    'claude-3-haiku-20240307',
    'claude-3-opus-20240229',
]

# function calling supporting models
FUNCTION_CALLING_SUPPORTED_MODELS = [
    'claude-3-5-sonnet',
    'claude-3-5-sonnet-20240620',
    'claude-3-5-sonnet-20241022',
    'claude-3-5-haiku-20241022',
    'gpt-4o-mini',
    'gpt-4o',
]


class LLM(RetryMixin, DebugMixin):
    """The LLM class represents a Language Model instance.

    Attributes:
        config: an LLMConfig object specifying the configuration of the LLM.
    """

    def __init__(
        self,
        config: LLMConfig,
        metrics: Metrics | None = None,
    ):
        """Initializes the LLM. If LLMConfig is passed, its values will be the fallback.

        Passing simple parameters always overrides config.

        Args:
            config: The LLM configuration.
            metrics: The metrics to use.
        """
        self._tried_model_info = False
        self.metrics: Metrics = (
            metrics if metrics is not None else Metrics(model_name=config.model)
        )
        self.cost_metric_supported: bool = True
        self.config: LLMConfig = copy.deepcopy(config)

        # litellm actually uses base Exception here for unknown model
        self.model_info: ModelInfo | None = None

        if self.config.log_completions:
            if self.config.log_completions_folder is None:
                raise RuntimeError(
                    'log_completions_folder is required when log_completions is enabled'
                )
            os.makedirs(self.config.log_completions_folder, exist_ok=True)

        # Initialize tokenizer for token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Claude-3's tokenizer

        self._completion = partial(
            litellm_completion,
            model=self.config.model,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            api_version=self.config.api_version,
            custom_llm_provider=self.config.custom_llm_provider,
            max_tokens=self.config.max_output_tokens,
            timeout=self.config.timeout,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            drop_params=self.config.drop_params,
        )

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.init_model_info()
        if self.vision_is_active():
            logger.debug('LLM: model has vision enabled')
        if self.is_caching_prompt_active():
            logger.debug('LLM: caching prompt enabled')
        if self.is_function_calling_active():
            logger.debug('LLM: model supports function calling')

        self._completion = partial(
            litellm_completion,
            model=self.config.model,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            api_version=self.config.api_version,
            custom_llm_provider=self.config.custom_llm_provider,
            max_tokens=self.config.max_output_tokens,
            timeout=self.config.timeout,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            drop_params=self.config.drop_params,
        )

        self._completion_unwrapped = self._completion

        @self.retry_decorator(
            num_retries=self.config.num_retries,
            retry_exceptions=LLM_RETRY_EXCEPTIONS,
            retry_min_wait=self.config.retry_min_wait,
            retry_max_wait=self.config.retry_max_wait,
            retry_multiplier=self.config.retry_multiplier,
        )
        def wrapper(*args, **kwargs):
            """Wrapper for the litellm completion function. Logs the input and output of the completion function."""
            from openhands.core.utils import json

            messages: list[dict[str, Any]] | dict[str, Any] = []
            mock_function_calling = kwargs.pop('mock_function_calling', False)

            # some callers might send the model and messages directly
            # litellm allows positional args, like completion(model, messages, **kwargs)
            if len(args) > 1:
                # ignore the first argument if it's provided (it would be the model)
                # design wise: we don't allow overriding the configured values
                # implementation wise: the partial function set the model as a kwarg already
                # as well as other kwargs
                messages = args[1] if len(args) > 1 else args[0]
                kwargs['messages'] = messages

                # remove the first args, they're sent in kwargs
                args = args[2:]
            elif 'messages' in kwargs:
                messages = kwargs['messages']

            # ensure we work with a list of messages
            messages = messages if isinstance(messages, list) else [messages]

            # Count total tokens in messages
            total_tokens = 0
            for msg in messages:
                if isinstance(msg.get('content'), str):
                    total_tokens += len(self.tokenizer.encode(msg['content']))

            # Check against max_input_tokens limit
            if total_tokens > self.config.max_input_tokens:
                raise ValueError(f"Input tokens ({total_tokens}) exceed max_input_tokens limit ({self.config.max_input_tokens})")

            original_fncall_messages = copy.deepcopy(messages)
            mock_fncall_tools = None
            if mock_function_calling:
                assert (
                    'tools' in kwargs
                ), "'tools' must be in kwargs when mock_function_calling is True"
                messages = convert_fncall_messages_to_non_fncall_messages(
                    messages, kwargs['tools']
                )
                kwargs['messages'] = messages
                kwargs['stop'] = STOP_WORDS
                mock_fncall_tools = kwargs.pop('tools')

            # if we have no messages, something went very wrong
            if not messages:
                raise ValueError(
                    'The messages list is empty. At least one message is required.'
                )

            # log the entire LLM prompt
            self.log_prompt(messages)

            if self.is_caching_prompt_active():
                # Anthropic-specific prompt caching
                if 'claude-3' in self.config.model:
                    kwargs['extra_headers'] = {
                        'anthropic-beta': 'prompt-caching-2024-07-31',
                    }

            try:
                # we don't support streaming here, thus we get a ModelResponse
                resp: ModelResponse = self._completion_unwrapped(*args, **kwargs)

                non_fncall_response = copy.deepcopy(resp)
                if mock_function_calling:
                    assert len(resp.choices) == 1
                    assert mock_fncall_tools is not None
                    non_fncall_response_message = resp.choices[0].message
                    fn_call_messages_with_response = (
                        convert_non_fncall_messages_to_fncall_messages(
                            messages + [non_fncall_response_message], mock_fncall_tools
                        )
                    )
                    fn_call_response_message = fn_call_messages_with_response[-1]
                    if not isinstance(fn_call_response_message, LiteLLMMessage):
                        fn_call_response_message = LiteLLMMessage(
                            **fn_call_response_message
                        )
                    resp.choices[0].message = fn_call_response_message

                # log for evals or other scripts that need the raw completion
                if self.config.log_completions:
                    assert self.config.log_completions_folder is not None
                    log_file = os.path.join(
                        self.config.log_completions_folder,
                        # use the metric model name (for draft editor)
                        f'{self.metrics.model_name.replace("/", "__")}-{time.time()}.json',
                    )

                    _d = {
                        'messages': messages,
                        'response': resp,
                        'args': args,
                        'kwargs': {k: v for k, v in kwargs.items() if k != 'messages'},
                        'timestamp': time.time(),
                        'cost': self._completion_cost(resp),
                    }
                    if mock_function_calling:
                        # Overwrite response as non-fncall to be consistent with `messages``
                        _d['response'] = non_fncall_response
                        # Save fncall_messages/response separately
                        _d['fncall_messages'] = original_fncall_messages
                        _d['fncall_response'] = resp
                    with open(log_file, 'w') as f:
                        f.write(json.dumps(_d))

                message_back: str = resp['choices'][0]['message']['content']

                # log the LLM response
                self.log_response(message_back)

                # post-process the response
                self._post_completion(resp)

                return resp
            except APIError as e:
                if 'Attention Required! | Cloudflare' in str(e):
                    raise CloudFlareBlockageError(
                        'Request blocked by CloudFlare'
                    ) from e
                raise

        self._completion = wrapper

    @property
    def completion(self):
        """Decorator for the litellm completion function.

        Check the complete documentation at https://litellm.vercel.app/docs/completion
        """
        return self._completion

    def init_model_info(self):
        if self._tried_model_info:
            return
        self._tried_model_info = True
        try:
            if self.config.model.startswith('openrouter'):
                self.model_info = litellm.get_model_info(self.config.model)
                # OpenRouter returns max_tokens but not max_input_tokens
                if (
                    self.model_info 
                    and 'max_tokens' in self.model_info 
                    and isinstance(self.model_info['max_tokens'], int)
                ):
                    # Set max_input_tokens to 80% of max_tokens to leave room for output
                    self.config.max_input_tokens = int(self.model_info['max_tokens'] * 0.8)
        except Exception as e:
            logger.debug(f'Error getting model info: {e}')

        if self.config.model.startswith('litellm_proxy/'):
            # IF we are using LiteLLM proxy, get model info from LiteLLM proxy
            # GET {base_url}/v1/model/info with litellm_model_id as path param
            response = requests.get(
                f'{self.config.base_url}/v1/model/info',
                headers={'Authorization': f'Bearer {self.config.api_key}'},
            )
            resp_json = response.json()
            if 'data' not in resp_json:
                logger.error(
                    f'Error getting model info from LiteLLM proxy: {resp_json}'
                )
            all_model_info = resp_json.get('data', [])
            current_model_info = next(
                (
                    info
                    for info in all_model_info
                    if info['model_name']
                    == self.config.model.removeprefix('litellm_proxy/')
                ),
                None,
            )
            if current_model_info:
                self.model_info = current_model_info['model_info']

        # Last two attempts to get model info from NAME
        if not self.model_info:
            try:
                self.model_info = litellm.get_model_info(
                    self.config.model.split(':')[0]
                )
            # noinspection PyBroadException
            except Exception:
                pass
        if not self.model_info:
            try:
                self.model_info = litellm.get_model_info(
                    self.config.model.split('/')[-1]
                )
            # noinspection PyBroadException
            except Exception:
                pass
        logger.debug(f'Model info: {self.model_info}')

        if self.config.model.startswith('huggingface'):
            # HF doesn't support the OpenAI default value for top_p (1)
            logger.debug(
                f'Setting top_p to 0.9 for Hugging Face model: {self.config.model}'
            )
            self.config.top_p = 0.9 if self.config.top_p == 1 else self.config.top_p

        # Set the max tokens in an LM-specific way if not set
        if self.config.max_input_tokens is None:
            if (
                self.model_info is not None
                and 'max_input_tokens' in self.model_info
                and isinstance(self.model_info['max_input_tokens'], int)
            ):
                self.config.max_input_tokens = self.model_info['max_input_tokens']
            else:
                # Safe fallback for any potentially viable model
                self.config.max_input_tokens = 4096

        if self.config.max_output_tokens is None:
            # Safe default for any potentially viable model
            self.config.max_output_tokens = 4096
            if self.model_info is not None:
                # max_output_tokens has precedence over max_tokens, if either exists.
                # litellm has models with both, one or none of these 2 parameters!
                if 'max_output_tokens' in self.model_info and isinstance(
                    self.model_info['max_output_tokens'], int
                ):
                    self.config.max_output_tokens = self.model_info['max_output_tokens']
                elif 'max_tokens' in self.model_info and isinstance(
                    self.model_info['max_tokens'], int
                ):
                    self.config.max_output_tokens = self.model_info['max_tokens']

    def vision_is_active(self) -> bool:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return not self.config.disable_vision and self._supports_vision()

    def _supports_vision(self) -> bool:
        """Acquire from litellm if model is vision capable.

        Returns:
            bool: True if model is vision capable. Return False if model not supported by litellm.
        """
        # litellm.supports_vision currently returns False for 'openai/gpt-...' or 'anthropic/claude-...' (with prefixes)
        # but model_info will have the correct value for some reason.
        # we can go with it, but we will need to keep an eye if model_info is correct for Vertex or other providers
        # remove when litellm is updated to fix https://github.com/BerriAI/litellm/issues/5608
        # Check both the full model name and the name after proxy prefix for vision support
        return (
            litellm.supports_vision(self.config.model)
            or litellm.supports_vision(self.config.model.split('/')[-1])
            or (
                self.model_info is not None
                and self.model_info.get('supports_vision', False)
            )
        )

    def is_caching_prompt_active(self) -> bool:
        """Check if prompt caching is supported and enabled for the current model."""
        if not self.config.caching_prompt:
            return False
        if self.config.model.startswith('claude-3'):
            model_name = self.config.model.split('/')[-1]
            return model_name in CACHE_PROMPT_SUPPORTED_MODELS
        return False

    def is_function_calling_active(self) -> bool:
        """Check if function calling is supported for the current model."""
        if self.config.model.startswith('claude-3'):
            model_name = self.config.model.split('/')[-1]
            return model_name in FUNCTION_CALLING_SUPPORTED_MODELS
        return False

    def _completion_cost(self, resp: ModelResponse) -> float | None:
        """Calculate the cost of a completion."""
        try:
            cost = litellm_completion_cost(
                completion_response=resp,
                model=self.config.model,
                custom_llm_provider=self.config.custom_llm_provider,
            )
            if cost is not None:
                self.metrics.add_cost(cost)
            return cost
        except Exception as e:
            logger.debug(f'Error calculating completion cost: {e}')
            self.cost_metric_supported = False
            return None

    def _post_completion(self, resp: ModelResponse):
        """Post-process a completion response."""
        if self.cost_metric_supported:
            self._completion_cost(resp)

    def reset(self):
        """Reset the LLM instance."""
        self.metrics = Metrics(model_name=self.metrics.model_name)
