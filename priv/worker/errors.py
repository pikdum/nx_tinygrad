"""Structured worker errors.

Each maps to an error `class` string in the protocol response. The Elixir side
turns these back into typed exceptions.
"""
from __future__ import annotations


class WorkerError(Exception):
    """Base class for all structured worker errors."""

    error_class = "WorkerError"

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ProtocolError(WorkerError):
    error_class = "ProtocolError"


class UnsupportedOperation(WorkerError):
    error_class = "UnsupportedOperation"


class UnsupportedDtype(WorkerError):
    error_class = "UnsupportedDtype"


class GraphValidationError(WorkerError):
    error_class = "GraphValidationError"


class DeviceUnavailable(WorkerError):
    error_class = "DeviceUnavailable"


class StaleReference(WorkerError):
    error_class = "StaleReference"


class CompileError(WorkerError):
    error_class = "CompileError"
