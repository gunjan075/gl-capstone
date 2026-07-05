"""Insurance price prediction - production pipeline package.

End-to-end regression workflow for estimating ``insurance_cost`` from health,
lifestyle, habit, and demographic variables. Notebooks in ``experiments/`` and
the serving layers (FastAPI, Streamlit) import from this package so that the
experiment and production tracks stay in sync.
"""

__version__ = "0.1.0"
