import importlib
pkgs = ["tensorflow", "sklearn", "pandas", "numpy", "joblib", "matplotlib"]
for p in pkgs:
    try:
        m = importlib.import_module(p)
        print(f"OK  {p}: {getattr(m, '__version__', 'n/a')}")
    except ImportError as e:
        print(f"MISSING {p}: {e}")
