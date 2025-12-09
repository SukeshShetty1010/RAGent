"""
agent/base.py

Defines the abstract Tool interface that all agent tooling must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class Tool(ABC):
    """
    Abstract base class / interface for tools used by the agent.

    Attributes
    ----------
    name : str
        Unique tool name.
    description : str
        Human-readable description of the tool's purpose.
    """

    name: str
    description: str

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, args: Dict[str, Any]) -> Any:
        """
        Execute the tool using a dictionary of arguments.

        Parameters
        ----------
        args : Dict[str, Any]
            A mapping of argument names to values. Concrete tools define
            required/optional keys.

        Returns
        -------
        Any
            Tool-specific result.
        """
        raise NotImplementedError()
