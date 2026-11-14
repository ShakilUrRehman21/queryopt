"""
executor — Volcano (iterator) query execution engine for QueryOpt Lab.

Implements the pull-based iterator model:
  open()  — allocate resources, initialize state
  next()  — pull the next row (returns None when stream is exhausted)
  close() — release resources, clean up state
"""
