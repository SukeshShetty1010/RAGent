# agent/tools/registry.py
"""
Tool registry for the agent.

Provides a simple catalog to register and look up Tool instances by name.
"""

from __future__ import annotations

from typing import Dict, List
import logging

from agent.base import Tool

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


class ToolRegistry:
    """
    Central registry for Tool instances.

    Example:
        registry = ToolRegistry()
        registry.register(my_tool)
        tool = registry.get("my_tool_name")
    """

    def __init__(self) -> None:
        # private storage mapping tool.name -> Tool instance
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """
        Register a Tool instance in the registry.

        Parameters
        ----------
        tool : Tool
            The tool instance to register. The tool's `name` attribute will be the key.
        """
        if not isinstance(tool, Tool):
            raise TypeError("register() expects a Tool instance")
        key = tool.name
        if not key or not isinstance(key, str):
            raise ValueError("Tool.name must be a non-empty string")
        prev = self._tools.get(key)
        self._tools[key] = tool
        if prev is None:
            logger.info("Registered tool: %s", key)
        else:
            logger.info("Re-registered tool (overwrote existing): %s", key)

    def get(self, name: str) -> Tool:
        """
        Retrieve a registered Tool by name.

        Parameters
        ----------
        name : str
            The tool name to look up.

        Returns
        -------
        Tool
            The registered tool instance.

        Raises
        ------
        KeyError
            If no tool with the given name is registered.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"Tool not found in registry: '{name}'")
        return tool

    def list_tools(self) -> List[Dict[str, str]]:
        """
        List metadata for all registered tools.

        Returns
        -------
        List[Dict[str, str]]
            A list of dicts with keys "name" and "description".
        """
        out = []
        for name, tool in self._tools.items():
            desc = getattr(tool, "description", "") or ""
            out.append({"name": name, "description": desc})
        return out

    def exists(self, name: str) -> bool:
        """
        Helper to check whether a tool is registered.

        Parameters
        ----------
        name : str

        Returns
        -------
        bool
        """
        return bool(name and name in self._tools)
