import os
import sys
from typing import Dict, Tuple, List, Optional
import copy
import logging

sys.path.append('..')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import PREFIX_MULTI_AGENTS, load_config
from agents import Reader, Planner, Developer, Reviewer, Summarizer
from state import State

class SOP:
    def __init__(self, competition: str, model: str):
        self.competition = competition
        self.model = model.replace("_", "-") if model else "config"
        self.state_records = []
        self.current_state = None
        self.config = self._load_configuration()
        
    def _load_configuration(self) -> Dict:
        config = load_config(f'{PREFIX_MULTI_AGENTS}/config.json')
        return {
            'max_iterations': 3,
            'phases': config['phases'],
            'phase_to_iterations': config['phase_to_iterations'],
            'llm_models': config.get('llm_models', {})
        }

    def _model_for_agent(self, agent_name: str) -> str:
        env_name = f"AUTOKAGGLE_{agent_name.upper()}_MODEL"
        env_model = os.getenv(env_name)
        if env_model:
            return env_model.replace("_", "-")

        configured_model = self.config['llm_models'].get(agent_name)
        if configured_model:
            return configured_model.replace("_", "-")

        defaults = {
            "Reader": "gpt-4o-mini",
            "Planner": "gpt-4o",
            "Developer": "gpt-4o",
            "Reviewer": "gpt-4o-mini",
            "Summarizer": "gpt-4o-mini",
        }
        return defaults[agent_name]

    def _create_agent(self, agent_name: str):
        model = self._model_for_agent(agent_name)
        if agent_name == "Planner" and self.model not in {"config", "default"}:
            model = self.model

        if agent_name == "Reader":
            agent = Reader(model, 'api')
        elif agent_name == "Planner":
            agent = Planner(model, 'api')
        elif agent_name == "Developer":
            agent = Developer(model, 'api')
        elif agent_name == "Reviewer":
            agent = Reviewer(model, 'api')
        elif agent_name == "Summarizer":
            agent = Summarizer(model, 'api')
        else:
            return None
        return agent

    def step(self, state: State) -> Tuple[str, State]:
        logging.info(f"Current State: {state}")
        state.make_dir()
        state.make_context()
        
        while not state.finished:
            current_agent_name = state.get_current_agent()
            current_agent = self._create_agent(current_agent_name)
            
            if current_agent is None:
                raise ValueError(f"Unknown agent: {current_agent_name}")
            
            action_result = current_agent.action(state)
            state.update_memory(action_result)
            state.next_step()

            if state.check_finished():
                state.set_score()
                exec_state_info, new_state = self.update_state(state)
                if exec_state_info == 'Success':
                    state.restore_memory()
        
        return exec_state_info, new_state

    def update_state(self, state: State) -> Tuple[str, Optional[State]]:
        self.state_records.append(copy.deepcopy(state))
        
        if state.phase == "Model Building, Validation, and Prediction":
            return self._update_model_building_state(state)
        else:
            return self._update_other_state(state)

    def _update_model_building_state(self, state: State) -> Tuple[str, Optional[State]]:
        if state.score < 3 and self.config['phase_to_iterations'][state.phase] < self.config['max_iterations']:
            self.config['phase_to_iterations'][state.phase] += 1
            return "Repeat", self._create_repeat_state(state)
        elif state.score >= 3:
            return "Complete", None
        else:
            return "Fail", None

    def _update_other_state(self, state: State) -> Tuple[str, Optional[State]]:
        if state.phase == "Feature Engineering":
            if len(self.state_records) < 2 or self.state_records[-2].phase != "Model Building, Validation, and Prediction":
                self.config['phase_to_iterations'][state.phase] += 1
        else:
            self.config['phase_to_iterations'][state.phase] += 1
        
        if state.score < 3 and self.config['phase_to_iterations'][state.phase] < self.config['max_iterations']:
            return "Repeat", self._create_repeat_state(state)
        elif state.score >= 3:
            next_phase = self.get_next_phase(state.phase)
            return "Success", State(phase=next_phase, competition=self.competition)
        else:
            return "Fail", None

    def _create_repeat_state(self, state: State) -> State:
        new_state = State(phase=state.phase, competition=self.competition)
        new_state.memory = copy.deepcopy(state.memory)
        new_state.memory.append({})
        return new_state

    def get_next_phase(self, current_phase: str) -> str:
        phases = self.config['phases']
        next_index = phases.index(current_phase) + 1
        return phases[next_index] if next_index < len(phases) else "Complete"
