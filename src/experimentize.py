# v0.8

from inspect import signature as _signature
import types as _types
import re as _re
import pandas as _pd
import traceback as _traceback
import os as _os

__all__ = ['param', 'experimentize', 'AsExperiment', 'ParamsBase']

class _ParamMarker:
    @staticmethod
    def valid(value):
        return isinstance(value, _ParamMarker)

    def __init__(self, config=None):
        self.config = config or {
            'as_func': None
        }
    
    def __call__(self, *, as_func=None):
        return _ParamMarker({
            **self.config,
            **{k: v for k, v in locals().items() if v is not None}
        })
    
    def _transform(self, *, name, value, target):
        if self.config['as_func'] is not None:
            func_name = self.config['as_func'].format(_re.sub("\\W", "", _re.sub("\s+|[_\-]", "_", str(value).lower())))
            if not hasattr(target, func_name):
                raise Exception(f'Function {func_name} is not defined (derived from {name}="{value}")')
            return getattr(target, func_name)
        return value
param = _ParamMarker()
        


def getall(obj):
    return {k: getattr(obj, k) for k in dir(obj)}

class ParamsBase:
    def __init__(self, **kwargs):
        default_values = self._get_default_values()
        for k, v in default_values.items():
            setattr(self, k, v)

        generators = self._get_generators()
        for k, v in generators.items():
            setattr(self, k, v())
        
        for k, v in kwargs.items():
            if k not in default_values and k not in generators:
                raise ValueError(f"Unknown parameter {k}")
            setattr(self, k, v)
        self._diff = kwargs

    def clone(self):
        generators = self._get_generators()
        clean_values = {
            k: v for k, v in self._get_values().items() 
            if k not in generators or k in self._diff
        }
        return self.__class__(**clean_values)

    def __repr__(self):
        return f"Settings({', '.join(f'{k}={v}' for k, v in self._get_values().items())})"

    def _get_default_values(self):
        return {
            k: v for k, v in getall(self.__class__).items() 
            if not isinstance(v, (_types.FunctionType, _types.MethodType, staticmethod))
            and not k.startswith('_')
        }

    def _get_generators(self):
        return {
            k: v.__func__ for k, v in vars(self.__class__).items() 
            if isinstance(v, staticmethod)
            and not k.startswith('_')
        }

    def _get_values(self):
        return {
            k: v for k, v in getall(self).items() 
            if not isinstance(v, (_types.FunctionType, _types.MethodType))
            and not k.startswith('_')
        }

def experimentize(default: ParamsBase):
    if not issubclass(default, ParamsBase):
        raise Exception(f"Experimentize must be initialized with a parameter-defining class (got {default.__name__}, which is not a subclass of ParamsBase)")
    
    def inner(target):
        scope = _ExperimentScope(default)
        
        def function_wrapper(func):
            params_to_consume = {k: v.default for k, v in _signature(func).parameters.items() if _ParamMarker.valid(v.default)}
            def wrapper(self, *args, **kwargs):
                wrapped_func = lambda *args_inner, **kwargs_inner: func(self, *args_inner, **{
                    **{k: v._transform(
                        name=k,
                        value=scope.get(k),
                        target=self
                    ) for k, v in params_to_consume.items()}, 
                    **kwargs_inner
                })
                if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], AsExperiment):
                    return args[0].call(wrapped_func, scope)
                return wrapped_func(*args, **kwargs)
            return wrapper
        
        funcs = {
            k: v for k, v in getall(target).items() 
            if isinstance(v, (_types.FunctionType, _types.MethodType))
            and not k.startswith('__')
        }
        for k, v in funcs.items():
            setattr(target, k, function_wrapper(v))

        return target
    return inner

class AsExperiment:
    def __init__(self, grid, *, repetitions=1, with_cache=None):
        self.grid = grid
        self.repetitions = repetitions
        self.with_cache = with_cache

    def call(self, func, scope):
        def wrapper(*args, **kwargs):
            total_results = _pd.DataFrame()
            if self.with_cache is not None:
                try:
                    total_results = _pd.read_pickle(self.with_cache)
                except FileNotFoundError:
                    pass
            
            if self.with_cache is not None:
                _os.makedirs(_os.path.dirname(self.with_cache), exist_ok=True)
            for keys in self.iterate_experiments(scope):
                try:
                    if self.with_cache is not None and self.check_if_present(total_results, keys):
                        continue

                    results = func(*args, **kwargs)
                    results = self._convert_to_dataframe(results, keys)
                    total_results = _pd.concat([total_results, results], ignore_index=True)
                    if self.with_cache is not None:
                        total_results.to_pickle(self.with_cache)
                except Exception:
                    traceback = _traceback.format_exc().splitlines()
                    traceback = traceback[:1] + traceback[5:-1] + [f"\033[91m{traceback[-1]}\033[0m"]
                    traceback = "\n".join(traceback)
                    print(f"\033[91mError occured during experiment {keys}:\033[0m\n{traceback}\n")
                
            return self._wrap_results(total_results)
        return wrapper
    
    def iterate_experiments(self, scope):
        for params in self.grid:
            for i in range(self.repetitions):
                scope.activate(params.clone())
                yield {'iteration': i, **params._diff}
                scope.deactivate()
    
    def check_if_present(self, results, keys):
        try:
            for k,v in keys.items():
                results = results[results[k] == v]
            return len(results) > 0
        except KeyError:
            return False

    @staticmethod
    def _convert_to_dataframe(results, keys):
        if results is None:
            return _pd.DataFrame([keys])
        if isinstance(results, _pd.DataFrame):
            for k, v in keys.items():
                results[k] = v
            return results

        return _pd.DataFrame([{**keys,**results}])
    
    @staticmethod
    def _wrap_results(results):
        setattr(results, 'prettify', lambda: results.rename(lambda x: x.replace("_", " ").title() if x.lower() == x else x, axis='columns'))
        return results

class _ExperimentScope:
    def __init__(self, default: ParamsBase = None):
        self.default = default
        self.deactivate()

    def activate(self, param: ParamsBase):
        self.param = param

    def deactivate(self):
        self.param = self.default()

    def get(self, name):
        if not hasattr(self.param, name):
            raise Exception(f"Trying to bind parameter {name} that is not defined in settings")
        return getattr(self.param, name)