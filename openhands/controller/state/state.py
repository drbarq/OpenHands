import base64
import pickle
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from openhands.controller.state.task import RootTask
from openhands.core.logger import openhands_logger as logger
from openhands.core.schema import AgentState
from openhands.events.action import (
    MessageAction,
)
from openhands.events.action.agent import AgentFinishAction
from openhands.events.event import Event, EventSource
from openhands.llm.metrics import Metrics
from openhands.storage.files import FileStore


class TrafficControlState(str, Enum):
    # default state, no rate limiting
    NORMAL = 'normal'

    # task paused due to traffic control
    THROTTLING = 'throttling'

    # traffic control is temporarily paused
    PAUSED = 'paused'


RESUMABLE_STATES = [
    AgentState.RUNNING,
    AgentState.PAUSED,
    AgentState.AWAITING_USER_INPUT,
    AgentState.FINISHED,
]


@dataclass
class State:
    """
    Represents the running state of an agent in the OpenHands system, saving data of its operation and memory.

    - Multi-agent/delegate state:
      - store the task (conversation between the agent and the user)
      - the subtask (conversation between an agent and the user or another agent)
      - global and local iterations
      - delegate levels for multi-agent interactions
      - almost stuck state

    - Running state of an agent:
      - current agent state (e.g., LOADING, RUNNING, PAUSED)
      - traffic control state for rate limiting
      - confirmation mode
      - the last error encountered

    - Data for saving and restoring the agent:
      - save to and restore from a session
      - serialize with pickle and base64

    - Save / restore data about message history
      - start and end IDs for events in agent's history
      - summaries and delegate summaries

    - Metrics:
      - global metrics for the current task
      - local metrics for the current subtask

    - Extra data:
      - additional task-specific data
    """

    root_task: RootTask = field(default_factory=RootTask)
    # global iteration for the current task
    iteration: int = 0
    # local iteration for the current subtask
    local_iteration: int = 0
    # max number of iterations for the current task
    max_iterations: int = 100
    confirmation_mode: bool = False
    history: list[Event] = field(default_factory=list)
    _llm = None  # Will be set when initializing with an agent
    preserve_last_n: int = field(default=10)  # Number of recent messages to preserve without summarization
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    agent_state: AgentState = AgentState.LOADING
    resume_state: AgentState | None = None
    traffic_control_state: TrafficControlState = TrafficControlState.NORMAL
    # global metrics for the current task
    metrics: Metrics = field(default_factory=Metrics)
    # local metrics for the current subtask
    local_metrics: Metrics = field(default_factory=Metrics)
    # root agent has level 0, and every delegate increases the level by one
    delegate_level: int = 0
    # start_id and end_id track the range of events in history
    start_id: int = -1
    end_id: int = -1
    # truncation_id tracks where to load history after context window truncation
    truncation_id: int = -1
    almost_stuck: int = 0
    delegates: dict[tuple[int, int], tuple[str, str]] = field(default_factory=dict)
    # NOTE: This will never be used by the controller, but it can be used by different
    # evaluation tasks to store extra data needed to track the progress/state of the task.
    extra_data: dict[str, Any] = field(default_factory=dict)
    last_error: str = ''

    def save_to_session(self, sid: str, file_store: FileStore):
        pickled = pickle.dumps(self)
        logger.debug(f'Saving state to session {sid}:{self.agent_state}')
        encoded = base64.b64encode(pickled).decode('utf-8')
        try:
            file_store.write(f'sessions/{sid}/agent_state.pkl', encoded)
        except Exception as e:
            logger.error(f'Failed to save state to session: {e}')
            raise e

    @staticmethod
    def restore_from_session(sid: str, file_store: FileStore) -> 'State':
        try:
            encoded = file_store.read(f'sessions/{sid}/agent_state.pkl')
            pickled = base64.b64decode(encoded)
            state = pickle.loads(pickled)
        except Exception as e:
            logger.warning(f'Could not restore state from session: {e}')
            raise e

        # update state
        if state.agent_state in RESUMABLE_STATES:
            state.resume_state = state.agent_state
        else:
            state.resume_state = None

        # first state after restore
        state.agent_state = AgentState.LOADING
        return state

    def __getstate__(self):
        # don't pickle history, it will be restored from the event stream
        state = self.__dict__.copy()
        state['history'] = []
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        # make sure we always have the attribute history
        if not hasattr(self, 'history'):
            self.history = []

    def get_current_user_intent(self) -> tuple[str | None, list[str] | None]:
        """Returns the latest user message and image(if provided) that appears after a FinishAction, or the first (the task) if nothing was finished yet."""
        last_user_message = None
        last_user_message_image_urls: list[str] | None = []
        for event in reversed(self.history):
            if isinstance(event, MessageAction) and event.source == 'user':
                last_user_message = event.content
                last_user_message_image_urls = event.image_urls
            elif isinstance(event, AgentFinishAction):
                if last_user_message is not None:
                    return last_user_message, None

        return last_user_message, last_user_message_image_urls

    def get_last_agent_message(self) -> MessageAction | None:
        for event in reversed(self.history):
            if isinstance(event, MessageAction) and event.source == EventSource.AGENT:
                return event
        return None

    def get_last_user_message(self) -> MessageAction | None:
        for event in reversed(self.history):
            if isinstance(event, MessageAction) and event.source == EventSource.USER:
                return event
        return None

    def set_llm(self, llm):
        """Set the LLM instance to use for token management."""
        self._llm = llm

    def append_event(self, event: Event) -> None:
        """Append an event to history with token management.
        
        This method will:
        1. Check if adding the event would exceed token limits
        2. If needed, summarize or truncate older history
        3. Update the history with the managed version
        """
        if not self._llm:
            # If no LLM is set, just append normally
            self.history.append(event)
            return

        # Use token management to get updated history
        self.history = self.manage_history_tokens(
            new_message=event,
            llm=self._llm,
            max_input_tokens=self._llm.config.max_input_tokens,
            preserve_last_n=self.preserve_last_n
        )

    def manage_history_tokens(
        self, 
        new_message: Event,
        llm,
        max_input_tokens: int | None = None,
        preserve_last_n: int | None = None
    ) -> list[Event]:
        """Manages history to keep token count within limits when adding a new message.
        
        Args:
            new_message: The new event to be added
            llm: The LLM instance to use for token counting and summarization
            max_input_tokens: Maximum allowed input tokens. If None, uses llm's config
            preserve_last_n: Number of most recent messages to preserve without summarization
            
        Returns:
            Updated history list with the new message, potentially truncated or summarized
        """
        if max_input_tokens is None:
            max_input_tokens = llm.config.max_input_tokens or 4096

        if preserve_last_n is None:
            preserve_last_n = self.preserve_last_n

        # Convert events to messages for token counting
        messages = []
        for event in self.history + [new_message]:
            if isinstance(event, MessageAction):
                messages.append({
                    "role": "user" if event.source == EventSource.USER else "assistant",
                    "content": event.content
                })

        try:
            # Get current token count
            token_count = llm.get_token_count(messages)
            
            if token_count <= max_input_tokens:
                # If within limits, just add the new message
                return self.history + [new_message]

            # Keep the most recent messages without summarization
            recent_history = self.history[-preserve_last_n:] if len(self.history) > preserve_last_n else self.history
            older_history = self.history[:-preserve_last_n] if len(self.history) > preserve_last_n else []

            if not older_history:
                # If we only have recent history and it's still too long
                logger.warning("Recent messages alone exceed token limit. Forced to truncate recent history.")
                return recent_history[-5:] + [new_message]  # Keep last 5 messages as absolute minimum

            # Create a summary of older messages
            older_messages = []
            for event in older_history:
                if isinstance(event, MessageAction):
                    older_messages.append({
                        "role": "user" if event.source == EventSource.USER else "assistant",
                        "content": event.content
                    })

            summary_prompt = {
                "role": "user",
                "content": "Please provide a very concise summary of this conversation, focusing only on the most important points and decisions made:"
            }
            
            try:
                # Get summary from LLM
                summary_response = llm.completion(messages=older_messages + [summary_prompt])
                summary_content = summary_response['choices'][0]['message']['content']
                
                # Create a summary event
                summary_event = MessageAction(
                    content=f"[HISTORY SUMMARY: {summary_content}]",
                    source=EventSource.SYSTEM
                )
                
                # Combine summary with recent history and new message
                updated_history = [summary_event] + recent_history + [new_message]
                
                # Verify the new history fits within token limits
                updated_messages = []
                for event in updated_history:
                    if isinstance(event, MessageAction):
                        updated_messages.append({
                            "role": "user" if event.source == EventSource.USER else "assistant",
                            "content": event.content
                        })
                
                final_token_count = llm.get_token_count(updated_messages)
                if final_token_count <= max_input_tokens:
                    self.truncation_id = len(self.history) - preserve_last_n
                    return updated_history
                
                # If still too long, fall back to simple truncation
                logger.warning("Summary still exceeds token limit. Falling back to truncation.")
                return recent_history[-5:] + [new_message]
                
            except Exception as e:
                logger.error(f"Failed to create history summary: {e}")
                # Fall back to simple truncation if summarization fails
                return recent_history + [new_message]
                
        except Exception as e:
            logger.error(f"Error in token management: {e}")
            # If token counting fails, be conservative and keep only recent history
            return (self.history[-preserve_last_n:] if len(self.history) > preserve_last_n else self.history) + [new_message]
