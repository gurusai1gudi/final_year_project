import importlib
def try_import(name):
    try:
        m = importlib.import_module(name)
        return getattr(m, '__version__', getattr(m, 'get_version', lambda: 'unknown')())
    except Exception as e:
        return f'not installed ({e.__class__.__name__})'

print('xgboost:', try_import('xgboost'))
print('django:', try_import('django'))
print('pandas:', try_import('pandas'))
